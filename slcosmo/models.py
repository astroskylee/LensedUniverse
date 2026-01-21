import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import arviz as az

from .tools import tool


class SLCOSMO:
    def __init__(self):
        # Cosmological parameters (remove if not needed).
        self.Omegam_true = 0.3
        self.Omegak_true = 0.0  # Flat universe
        self.w0_true = -1.0
        self.wa_true = 0.0

    def run_inference(
        self,
        data_dict,
        sampler_type,
        cosmology_type="wcdm",
        cosmo_prior=None,
        sample_h0=False,
        num_warmup=500,
        num_samples=2000,
        num_chains=10,
        jax_key=0,
    ):
        # Select models from data_dict keys.
        selected_models = list(data_dict.keys())

        sampler = MCMC(
            sampler_type,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
            chain_method="vectorized",
        )

        print(f'Running inference with models: {"+".join(selected_models)}')
        sampler.run(
            jax.random.PRNGKey(jax_key),
            selected_models=selected_models,
            data_dict=data_dict,
            cosmology_type=cosmology_type,
            cosmo_prior=cosmo_prior,
            sample_h0=sample_h0,
        )

        inf_data = az.from_numpyro(sampler)
        return inf_data


class SLmodel:
    def __init__(self, slcosmo):
        self.slcosmo = slcosmo
        self.c = 299792.458  # km/s

        # Map model names to model methods.
        self.model_dict = {
            "dspl": self.DSPL_model,
            "dspl_mst": self.DSPL_model_mst,
            "dspl_mst_bias": self.DSPL_model_mst_bias,
            "sne": self.LensedSNe_model,
            "sne_bias": self.LensedSNe_model_mst_bias,
            "kinematic": self.lens_kinematic_model,
            "kinematic_bias": self.lens_kinematic_model_bias,
            "kinematic_gamma": self.lens_kinematic_gamma,
        }

    def cosmology_model(self, cosmology_type, cosmo_prior, sample_h0=False):
        """Define cosmology parameters."""

        cosmology = {
            "Omegam": numpyro.sample("Omegam", dist.Uniform(cosmo_prior["omegam_low"], cosmo_prior["omegam_up"])),
            "Omegak": 0.0,
            "w0": -1.0,
            "wa": 0.0,
            "h0": 70,
        }

        if cosmology_type == "wcdm":
            cosmology["w0"] = numpyro.sample("w0", dist.Uniform(cosmo_prior["w0_low"], cosmo_prior["w0_up"]))
        elif cosmology_type == "owcdm":
            cosmology["Omegak"] = numpyro.sample(
                "Omegak", dist.Uniform(cosmo_prior["omegak_low"], cosmo_prior["omegak_up"])
            )
            cosmology["w0"] = numpyro.sample("w0", dist.Uniform(cosmo_prior["w0_low"], cosmo_prior["w0_up"]))
        elif cosmology_type == "waw0cdm":
            cosmology["w0"] = numpyro.sample("w0", dist.Uniform(cosmo_prior["w0_low"], cosmo_prior["w0_up"]))
            cosmology["wa"] = numpyro.sample("wa", dist.Uniform(cosmo_prior["wa_low"], cosmo_prior["wa_up"]))
        elif cosmology_type == "owaw0cdm":
            cosmology["Omegak"] = numpyro.sample(
                "Omegak", dist.Uniform(cosmo_prior["omegak_low"], cosmo_prior["omegak_up"])
            )
            cosmology["w0"] = numpyro.sample("w0", dist.Uniform(cosmo_prior["w0_low"], cosmo_prior["w0_up"]))
            cosmology["wa"] = numpyro.sample("wa", dist.Uniform(cosmo_prior["wa_low"], cosmo_prior["wa_up"]))
        elif cosmology_type != "lambdacdm":
            raise ValueError(
                "Unknown cosmology_type: "
                f"{cosmology_type} available types: lambdacdm, wcdm, waw0cdm, owcdm, owaw0cdm"
            )

        if sample_h0:
            cosmology["h0"] = numpyro.sample("h0", dist.Uniform(cosmo_prior["h0_low"], cosmo_prior["h0_up"]))

        return cosmology

    def DSPL_model(self, dspl_data, cosmology):
        """DSPL model."""
        zl = dspl_data["zl_dspl"]
        zs1 = dspl_data["zs1_dspl"]
        zs2 = dspl_data["zs2_dspl"]
        beta_obs = dspl_data["beta_obs_dspl"]
        sigma_beta = dspl_data["sigma_beta_dspl"]

        Dl_dspl, Ds1_dspl, Dls1_dspl = tool.compute_distances(zl, zs1, cosmology)
        Dl_dspl, Ds2_dspl, Dls2_dspl = tool.compute_distances(zl, zs2, cosmology)

        beta_model_dspl = (Dls1_dspl * Ds2_dspl) / (Ds1_dspl * Dls2_dspl)
        with numpyro.plate("DSPL_data", len(zl)):
            numpyro.sample("beta_obs_dspl", dist.Normal(beta_model_dspl, sigma_beta), obs=beta_obs)

    def DSPL_model_mst(self, dspl_data, cosmology):
        zl = dspl_data["zl_dspl"]
        zs1 = dspl_data["zs1_dspl"]
        zs2 = dspl_data["zs2_dspl"]
        theta_1 = dspl_data["theta_1"]
        theta_2 = dspl_data["theta_2"]
        beta_obs = dspl_data["beta_obs_dspl"]
        sigma_beta = dspl_data["sigma_beta_dspl"]
        lambda_int = dspl_data["lambda_int"]
        lambda_int_err = dspl_data["lambda_int_err"]

        Dl_dspl, Ds1_dspl, Dls1_dspl = tool.compute_distances(zl, zs1, cosmology)
        Dl_dspl, Ds2_dspl, Dls2_dspl = tool.compute_distances(zl, zs2, cosmology)

        beta_mst = (Dls1_dspl * Ds2_dspl) / (Ds1_dspl * Dls2_dspl)
        with numpyro.plate("DSPL_data", len(zl)):
            mst = numpyro.sample("mst_dspl", dist.TruncatedNormal(lambda_int, lambda_int_err, low=0.5, high=1.5))
            beta = tool.beta_antimst(beta_mst, mst)
            numpyro.sample("beta_obs_dspl", dist.TruncatedNormal(beta, sigma_beta, low=0, high=1), obs=beta_obs)
        numpyro.deterministic("mean_mst", jnp.mean(mst))

    def DSPL_model_mst_bias(self, dspl_data, cosmology, bias):
        zl = dspl_data["zl_dspl"]
        zs1 = dspl_data["zs1_dspl"]
        zs2 = dspl_data["zs2_dspl"]
        theta_1 = dspl_data["theta_1"]
        theta_2 = dspl_data["theta_2"]
        beta_obs = dspl_data["beta_obs_dspl"]
        sigma_beta = dspl_data["sigma_beta_dspl"]
        lambda_int = dspl_data["lambda_int"]
        lambda_int_err = dspl_data["lambda_int_err"]

        Dl_dspl, Ds1_dspl, Dls1_dspl = tool.compute_distances(zl, zs1, cosmology)
        Dl_dspl, Ds2_dspl, Dls2_dspl = tool.compute_distances(zl, zs2, cosmology)

        beta_mst = (Dls1_dspl * Ds2_dspl) / (Ds1_dspl * Dls2_dspl)
        with numpyro.plate("DSPL_data", len(zl)):
            mst = numpyro.sample(
                "mst_dspl", dist.TruncatedNormal(lambda_int, lambda_int_err, low=0.3, high=1.8)
            ) + bias
            beta = tool.beta_antimst(beta_mst, mst)
            numpyro.sample("beta_obs_dspl", dist.TruncatedNormal(beta, sigma_beta, low=0, high=1), obs=beta_obs)
        numpyro.deterministic("mean_mst", jnp.mean(mst))

    def LensedSNe_model(self, sne_data, cosmology):
        """Lensed SNe model."""
        zl = sne_data["zl_sne"]
        zs = sne_data["zs_sne"]
        tmax = sne_data["tmax_sne"]
        Ddt_obs = sne_data["Ddt_obs_sne"]
        Ddt_err = sne_data["Ddt_err_sne"]

        Dl_sne, Ds_sne, Dls_sne = tool.compute_distances(zl, zs, cosmology)
        Ddt_sne = (1 + zl) * Dl_sne * Ds_sne / Dls_sne * self.c / cosmology["H0"] / 1000

        with numpyro.plate("LensedSNe_data", len(zl)):
            numpyro.sample("Ddt_obs_sne", dist.Normal(Ddt_sne, Ddt_err / 1000), obs=Ddt_obs / 1000)

    def LensedSNe_model_mst_bias(self, sne_data, cosmology, bias):
        """Lensed SNe model with MST bias."""
        zl = sne_data["zl_sne"]
        zs = sne_data["zs_sne"]
        tmax = sne_data["tmax_sne"]
        Ddt_obs = sne_data["Ddt_obs_sne"]
        Ddt_err = sne_data["Ddt_err_sne"]
        lambda_int = sne_data["lambda_int"]
        lambda_int_err = sne_data["lambda_int_err"]

        Dl_sne, Ds_sne, Dls_sne = tool.compute_distances(zl, zs, cosmology)
        Ddt_sne_mst = (1 + zl) * Dl_sne * Ds_sne / Dls_sne * self.c / cosmology["H0"] / 1000

        with numpyro.plate("LensedSNe_data", len(zl)):
            Ddt_sne = Ddt_sne_mst * (numpyro.sample("mst_sn", dist.Normal(lambda_int, lambda_int_err)) + bias)
            numpyro.sample("Ddt_obs_sne", dist.Normal(Ddt_sne, Ddt_err / 1000), obs=Ddt_obs / 1000)

    def lens_kinematic_model(self, kin_data, cosmology):
        """Lens kinematic model."""
        zl = kin_data["zl_kin"]
        zs = kin_data["zs_kin"]
        theta_E_obs = kin_data["theta_E_obs_kin"]
        delta = kin_data["delta_kin"]
        sigma_v_obs = kin_data["sigma_v_obs_kin"]
        vel_err = kin_data["vel_err_kin"]

        Dl_kin, Ds_kin, Dls_kin = tool.compute_distances(zl, zs, cosmology)

        # Extra priors
        gamma_mean = numpyro.sample("gamma", dist.Uniform(1.5, 2.5))
        gamma_sigma = numpyro.sample("gamma_sig", dist.TruncatedNormal(0.16, 1.0, low=0.0, high=0.4))
        beta_mean = numpyro.sample("beta_kin", dist.Uniform(-0.6, 1.0))
        beta_sigma = numpyro.sample("beta_sig_kin", dist.TruncatedNormal(0.13, 1.0, low=0.0, high=0.4))

        # Per-lens parameters
        with numpyro.plate("lens_kin_data", len(theta_E_obs)):
            y_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.1, high=2.5))
            beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-1.0, high=0.8))
            xsample = numpyro.sample("Ein_radius", dist.Normal(theta_E_obs, 0.01))

        fmass = tool.f_mass(y_i, delta, beta_i)

        # logfactor =  -300*(1-fmass>0)
        # numpyro.factor("cusp", logfactor )

        pre_vel = (
            jnp.sqrt(
                (3e8**2)
                / (2 * jnp.pi**0.5)
                * Ds_kin
                / Dls_kin
                * (xsample / 206265)
                * fmass
                * (0.725 / xsample) ** (2 - y_i)
            )
            / 1000
            / 100
        )

        with numpyro.plate("lens_kin_data_obs", len(theta_E_obs)):
            numpyro.sample("velocity_obs_kin", dist.Normal(pre_vel, vel_err / 100), obs=(sigma_v_obs / 100))

    def lens_kinematic_model_bias(self, kin_data, cosmology):
        """Lens kinematic model with MST bias."""
        zl = kin_data["zl_kin"]
        zs = kin_data["zs_kin"]
        theta_E_obs = kin_data["theta_E_obs_kin"]
        delta = kin_data["delta_kin"]
        sigma_v_obs = kin_data["sigma_v_obs_kin"]
        vel_err = kin_data["vel_err_kin"]

        Dl_kin, Ds_kin, Dls_kin = tool.compute_distances(zl, zs, cosmology)

        # Extra priors
        gamma_mean = numpyro.sample("gamma", dist.Uniform(1.5, 2.5))
        gamma_sigma = numpyro.sample("gamma_sig", dist.TruncatedNormal(0.16, 1.0, low=0.0, high=0.4))
        beta_mean = numpyro.sample("beta_kin", dist.Uniform(-0.6, 1.0))
        beta_sigma = numpyro.sample("beta_sig_kin", dist.TruncatedNormal(0.13, 1.0, low=0.0, high=0.4))
        lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.8, 1.2))
        lambda_sigma = numpyro.sample("lambda_mean_sig", dist.Uniform(0.0, 0.1))

        # Per-lens parameters
        with numpyro.plate("lens_kin_data", len(theta_E_obs)):
            lambda_npr = numpyro.sample("lambda", dist.Normal(lambda_mean, lambda_sigma))
            y_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.1, high=2.5))
            beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-1.0, high=0.8))
            xsample = numpyro.sample("Ein_radius", dist.Normal(theta_E_obs, 0.01))

        fmass = tool.f_mass(y_i, delta, beta_i)

        D_ratio = numpyro.deterministic("distance_ratio", Ds_kin / Dls_kin)
        pre_vel = (
            jnp.sqrt(lambda_npr)
            * jnp.sqrt(
                (3e8**2)
                / (2 * jnp.pi**0.5)
                * D_ratio
                * (xsample / 206265)
                * fmass
                * (0.725 / xsample) ** (2 - y_i)
            )
            / 1000
            / 100
        )

        with numpyro.plate("lens_kin_data_obs", len(theta_E_obs)):
            numpyro.sample("velocity_obs_kin", dist.Normal(pre_vel, vel_err / 100), obs=(sigma_v_obs / 100))

    def lens_kinematic_gamma(self, kin_data, cosmology):
        """Lens kinematic model with gamma population."""
        zl = kin_data["zl_kin"]
        zs = kin_data["zs_kin"]
        theta_E_obs = kin_data["theta_E_obs_kin"]
        delta = kin_data["delta_kin"]
        sigma_v_obs = kin_data["sigma_v_obs_kin"]
        sigma_v_true = kin_data["sigma_v_obs_kin"]
        vel_err = kin_data["vel_err_kin"]
        gamma_pop = kin_data["gamma_pop"]
        gamma_err = kin_data["gamma_err"]

        Dl_kin, Ds_kin, Dls_kin = tool.compute_distances(zl, zs, cosmology)

        beta_mean = numpyro.sample("beta_kin", dist.Uniform(-0.4, 0.6))
        beta_sigma = numpyro.sample("beta_sig_kin", dist.TruncatedNormal(0.13, 1.0, low=0.05, high=0.4))
        gamma_mean = numpyro.sample("gamma", dist.Uniform(1.5, 2.5))
        gamma_sigma = numpyro.sample("gamma_sig", dist.TruncatedNormal(0.16, 1.0, low=0.0, high=0.4))
        lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.8, 1.2))
        lambda_sigma = numpyro.sample("lambda_mean_sig", dist.Uniform(0.0, 0.1))
        # y_i = gamma_pop

        with numpyro.plate("lens_kin_data", len(theta_E_obs)):
            lambda_npr = numpyro.sample("lambda", dist.Normal(lambda_mean, lambda_sigma))
            y_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.1, high=2.7))
            beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-5, high=0.5))
            xsample = numpyro.sample("Ein_radius", dist.Normal(theta_E_obs, 0.01))

        fmass = tool.f_mass(y_i, delta, beta_i)
        D_ratio = numpyro.deterministic("distance_ratio", Ds_kin / Dls_kin)
        pre_vel = (
            jnp.sqrt(lambda_npr)
            * jnp.sqrt(
                (3e8**2)
                / (2 * jnp.pi**0.5)
                * D_ratio
                * (xsample / 206265)
                * fmass
                * (0.725 / xsample) ** (2 - y_i)
            )
            / 1000
            / 100
        )

        with numpyro.plate("lens_kin_data_obs", len(theta_E_obs)):
            numpyro.sample("gamma_obs", dist.Normal(y_i, gamma_err), obs=(gamma_pop))
            numpyro.sample("velocity_obs_kin", dist.Normal(pre_vel, vel_err / 100), obs=(sigma_v_obs / 100))

    def joint_model(self, selected_models, data_dict, cosmology_type="wcdm", sample_h0=False, cosmo_prior=None):
        """
        Joint model that dispatches selected submodels.

        Args:
            selected_models (list of str): Submodels to include, e.g. ["sne", "dspl"].
            data_dict (dict): Input data by submodel name.
            cosmology_type (str): Cosmology model type.
        """
        if cosmo_prior is None:
            cosmo_prior = {
                "w0_up": -0.5,
                "w0_low": -1.5,
                "wa_up": 2,
                "wa_low": -2,
                "omegak_up": 1,
                "omegak_low": -1,
                "h0_up": 80,
                "h0_low": 60,
                "omegam_up": 0.5,
                "omegam_low": 0.1,
            }

        cosmology = self.cosmology_model(cosmology_type, cosmo_prior, sample_h0=sample_h0)
        bias = numpyro.sample("bias", dist.TruncatedNormal(0, 1, low=-0.1, high=0.1))
        # Dispatch selected submodels
        for model_name in selected_models:
            if model_name in self.model_dict:
                if model_name == "dspl_mst_bias" or model_name == "sne_bias":
                    self.model_dict[model_name](data_dict[model_name], cosmology, bias)
                else:
                    self.model_dict[model_name](data_dict[model_name], cosmology)
