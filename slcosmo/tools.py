import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from cosmodj import (
    Planck18Cosmology,
    angular_diameter_distance as cosmodj_angular_diameter_distance,
    angular_diameter_distances as cosmodj_angular_diameter_distances,
    time_delay_distance as cosmodj_time_delay_distance,
)


class tool:
    c_km_s = 299792.458
    c_m_s = 299792458
    G = 6.6743e-11
    msun = 1.988409870698051e+30
    Mpc = 3.0856776e+22
    pc = 3.0856776e+16

    @staticmethod
    def _h0(cosmology):
        if cosmology is None:
            return Planck18Cosmology.H0
        if hasattr(cosmology, "H0"):
            return cosmology.H0
        if "H0" in cosmology:
            return cosmology["H0"]
        return cosmology["h0"]

    @staticmethod
    def angular_diameter_distance(z, cosmology, n=20):
        return cosmodj_angular_diameter_distance(z, cosmology, n=n)

    @staticmethod
    def dldsdls(zl, zs, cosmology, n=20):
        """
        Compute angular-diameter distances in Mpc.

        cosmology: dict, output of cosmology_model function.
        """
        return cosmodj_angular_diameter_distances(zl, zs, cosmology, n=n)

    @staticmethod
    def time_delay_distance(zl, zs, cosmology, n=20):
        """Compute lensing time-delay distance in Mpc."""
        return cosmodj_time_delay_distance(zl, zs, cosmology, n=n)

    def sigma_crit_jax(zl, zs, cosmology, n=20):
        # output is in Solar mass / pc**2
        Dl, Ds, Dls = tool.dldsdls(zl=zl, zs=zs, cosmology=cosmology, n=n)
        factor = tool.c_km_s ** 2 / (4.0 * jnp.pi * tool.G)
        return factor * (Ds / (Dl * Dls)) * tool.pc / tool.msun

    @staticmethod
    def compute_distances(zl, zs, cosmology, n=20):
        """
        Compute dimensionless angular-diameter distances.

        This preserves the historical slcosmo API used by the DSPL and model
        likelihood code. Use dldsdls() when physical Mpc distances are needed.

        cosmology: dict, output of cosmology_model function.
        """
        Dl, Ds, Dls = cosmodj_angular_diameter_distances(zl, zs, cosmology, n=n)
        scale = tool._h0(cosmology) / tool.c_km_s
        return Dl * scale, Ds * scale, Dls * scale

    @staticmethod
    def jgamma(n):
        return jnp.exp(gammaln(n))

    @staticmethod
    def f_mass(ygamma, delta, beta):
        eps = ygamma + delta - 2
        return (
            (3 - delta)
            / (eps - 2 * beta)
            / (3 - eps)
            * (
                tool.jgamma(eps / 2 - 1 / 2) / tool.jgamma(eps / 2)
                - beta * tool.jgamma(eps / 2 + 1 / 2) / tool.jgamma(eps / 2 + 1)
            )
            * tool.jgamma(ygamma / 2)
            * tool.jgamma(delta / 2)
            / (tool.jgamma(ygamma / 2 - 1 / 2) * tool.jgamma(delta / 2 - 1 / 2))
        )

    @staticmethod
    def FoM(samples_df, nbins=100, confidence=0.95):
        try:
            w0 = samples_df["w0"].values
            wa = samples_df["wa"].values if "wa" in samples_df.columns else np.zeros_like(w0)
        except Exception:
            w0 = samples_df.posterior["w0"].values
            wa = samples_df.posterior["wa"].values

        range_w0 = (np.min(w0), np.max(w0))
        range_wa = (np.min(wa), np.max(wa))
        H, w0_edges, wa_edges = np.histogram2d(w0, wa, bins=nbins, range=[range_w0, range_wa])
        H = H / np.sum(H)

        H_flat = H.flatten()
        idx_sorted = np.argsort(H_flat)[::-1]
        H_sorted = H_flat[idx_sorted]

        cumsum = np.cumsum(H_sorted)
        threshold_idx = np.where(cumsum >= confidence)[0][0]
        level_conf = H_sorted[threshold_idx]

        inside_bins = H >= level_conf
        bin_area = (w0_edges[1] - w0_edges[0]) * (wa_edges[1] - wa_edges[0])
        A_conf = np.sum(inside_bins) * bin_area

        FoM_val = (6.17 * np.pi) / A_conf
        return FoM_val

    @staticmethod
    def FoM_cov(samples_df):
        cov_matrix = samples_df[["w0", "wa"]].cov()
        det_cov = np.linalg.det(cov_matrix.values)
        return 1 / np.sqrt(det_cov)

    @staticmethod
    def inf2pd(inf_data):
        posterior = inf_data.posterior
        samples_dict = {}
        for var_name in posterior.data_vars:
            samples_dict[var_name] = posterior[var_name].values.flatten()
        return pd.DataFrame(samples_dict)
    ########################################################################################
    @staticmethod
    def beta_mst(beta, mst):
        eta = 1 / beta
        eta_mst = eta / (mst + (1 - mst) * eta)
        return 1 / eta_mst

    @staticmethod
    def beta_antimst(beta_mst, mst):
        eta_mst = 1 / beta_mst
        den = 1.0 - eta_mst * (1.0 - mst)
        eta = (eta_mst * mst) / den
        return 1 / eta

    @staticmethod
    def beta_mst_v3(beta, mst):
        eta = 1 / beta
        eta_mst = 1 - mst(1 - eta)
        return 1 / eta_mst

    @staticmethod
    def beta_antimst_v3(beta_mst, mst):
        eta_mst = 1 / beta_mst
        eta = 1 - (1 - eta_mst) / mst
        return 1 / eta
    ########################################################################################

    @staticmethod
    def EPL(R, thetaE, gamma):
        kappa = (3 - gamma) / 2 * (thetaE / R) ** (gamma - 1)
        return kappa

    @staticmethod
    def EPL_msunmpc(R, thetaE, gamma, zl, zs, cosmology):
        kappa = (3 - gamma) / 2 * (thetaE / R) ** (gamma - 1)
        return kappa * tool.sigma_crit_jax(zl=zl, zs=zs, cosmology=cosmology)

    def sersic_constant(sersic_index):
        # use less accurate one for direct comparison with lenstronomy
        bn = 1.9992 * sersic_index - 0.3271
        bn = jnp.maximum(bn, 0.00001)  # make sure bn is strictly positive as a save guard for very low n_sersic
        return bn

    def sersic_fn(r, mass_to_light_ratio, intensity, effective_radius, sersic_index, **_):
        b = tool.sersic_constant(sersic_index)
        r_ = (r / effective_radius) ** (1.0 / sersic_index)
        return mass_to_light_ratio * intensity * jnp.exp(-b * (r_ - 1.0))

    @staticmethod
    def truncated_normal(mean, std, low, high, size=None, random_state=None):
        """
        Draw samples from N(mean, std) truncated to [low, high].

        random_state can be:
          - None: use NumPy global RNG (legacy behavior; reproducible only if np.random.seed is set)
          - int: treated as a seed
          - np.random.Generator
          - np.random.RandomState
        """
        mean = np.asarray(mean)
        std = np.asarray(std)
        a = (low - mean) / std
        b = (high - mean) / std

        # Normalize random_state for SciPy
        if isinstance(random_state, (int, np.integer)):
            # For maximum SciPy compatibility, convert seed -> RandomState
            random_state = np.random.RandomState(int(random_state))

        return truncnorm(a, b, loc=mean, scale=std).rvs(size=size, random_state=random_state)

    def make_4d_interpolant(xg, yg, zg, wg, table):
        xg = jnp.asarray(xg)
        yg = jnp.asarray(yg)
        zg = jnp.asarray(zg)
        wg = jnp.asarray(wg)
        table = jnp.asarray(table)

        def find_idx_t(p, grid):
            """Return (index, weight) for linear interpolation along one dimension."""
            idx = jnp.searchsorted(grid, p, side="right") - 1
            idx = jnp.clip(idx, 0, grid.size - 2)
            t = (p - grid[idx]) / (grid[idx + 1] - grid[idx])
            return idx, t

        def interp_single(p1, p2, p3, p4):
            """Interpolate one single 4D point."""
            i1, t1 = find_idx_t(p1, xg)
            i2, t2 = find_idx_t(p2, yg)
            i3, t3 = find_idx_t(p3, zg)
            i4, t4 = find_idx_t(p4, wg)

            # Gather 16 corner values
            def T(a, b, c, d):
                return table[i1 + a, i2 + b, i3 + c, i4 + d]

            c = lambda a, b, c, d: T(a, b, c, d)

            return (
                c(0, 0, 0, 0) * (1 - t1) * (1 - t2) * (1 - t3) * (1 - t4)
                + c(0, 0, 0, 1) * (1 - t1) * (1 - t2) * (1 - t3) * (t4)
                + c(0, 0, 1, 0) * (1 - t1) * (1 - t2) * (t3) * (1 - t4)
                + c(0, 0, 1, 1) * (1 - t1) * (1 - t2) * (t3) * (t4)
                + c(0, 1, 0, 0) * (1 - t1) * (t2) * (1 - t3) * (1 - t4)
                + c(0, 1, 0, 1) * (1 - t1) * (t2) * (1 - t3) * (t4)
                + c(0, 1, 1, 0) * (1 - t1) * (t2) * (t3) * (1 - t4)
                + c(0, 1, 1, 1) * (1 - t1) * (t2) * (t3) * (t4)
                + c(1, 0, 0, 0) * (t1) * (1 - t2) * (1 - t3) * (1 - t4)
                + c(1, 0, 0, 1) * (t1) * (1 - t2) * (1 - t3) * (t4)
                + c(1, 0, 1, 0) * (t1) * (1 - t2) * (t3) * (1 - t4)
                + c(1, 0, 1, 1) * (t1) * (1 - t2) * (t3) * (t4)
                + c(1, 1, 0, 0) * (t1) * (t2) * (1 - t3) * (1 - t4)
                + c(1, 1, 0, 1) * (t1) * (t2) * (1 - t3) * (t4)
                + c(1, 1, 1, 0) * (t1) * (t2) * (t3) * (1 - t4)
                + c(1, 1, 1, 1) * (t1) * (t2) * (t3) * (t4)
            )

        # Batch version via vmap (safe for MCMC)
        interp_batch = jax.vmap(interp_single, in_axes=(0, 0, 0, 0))

        def interp_fun(p1, p2, p3, p4):
            """Dispatch scalar vs vector input automatically."""
            if jnp.ndim(p1) == 0:
                return interp_single(p1, p2, p3, p4)
            else:
                return interp_batch(p1, p2, p3, p4)

        return interp_fun
