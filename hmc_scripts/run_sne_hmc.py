#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import sys

workdir = Path(__file__).resolve().parent.parent
os.chdir(workdir)
sys.path.insert(0, str(workdir))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from jax import random
import arviz as az

from slcosmo.tools import tool
from hmc_scripts.corner_utils import select_corner_vars, make_overlay_corner

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()
if any(d.platform == "gpu" for d in jax.devices()):
    numpyro.set_platform("gpu")
else:
    numpyro.set_platform("cpu")

SEED = 42
rng_np = np.random.default_rng(SEED)
np.random.seed(SEED)

TEST_MODE = os.environ.get("COMBINE_FORECAST_TEST") == "1"
RESULT_DIR = Path("/mnt/lustre/tianli/LensedUniverse_result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = workdir / "result"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(os.environ.get("SLCOSMO_DATA_DIR", str(workdir / "data")))

cosmo_true = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}
cosmo_prior = {
    "w0_up": 0.0,   "w0_low": -2.0,
    "wa_up": 2.0,   "wa_low": -2.0,
    "omegak_up": 1.0, "omegak_low": -1.0,
    "h0_up": 80.0,  "h0_low": 60.0,
    "omegam_up": 0.5, "omegam_low": 0.1,
}

sn_data = pd.read_csv(DATA_DIR / "Euclid_150SNe.csv")
sn_data = sn_data[(sn_data["tmax"] >= 5) & (sn_data["tmax"] <= 80)]
sn_data = sn_data.nlargest(100, "tmax")

zl = sn_data["zl"].to_numpy()
zs = sn_data["z_host"].to_numpy()
t_delay_true_days = sn_data["tmax"].to_numpy()

zl_j = jnp.asarray(zl)
zs_j = jnp.asarray(zs)
Dl, Ds, Dls = tool.dldsdls(zl_j, zs_j, cosmo_true, n=20)
Ddt_geom = (1.0 + zl_j) * Dl * Ds / Dls
Ddt_geom = np.asarray(Ddt_geom)

c_km_day = tool.c_km_s * 86400.0
Mpc_km = tool.Mpc / 1000.0

sigma_t_days = 1.0
sigma_phi_frac = 0.04
sigma_lambda_frac = 0.08

lambda_pop_mean = 1.0
lambda_pop_sigma = 0.05
lambda_low, lambda_high = 0.8, 1.2
lambda_true = tool.truncated_normal(lambda_pop_mean, lambda_pop_sigma, lambda_low, lambda_high, len(zl), random_state=rng_np)

fermat_phi_true = (c_km_day * t_delay_true_days) / (Ddt_geom * Mpc_km)
fermat_phi_true = np.asarray(fermat_phi_true)

t_delay_true_mst = t_delay_true_days * lambda_true

# scale phi

def scale_phi(phi_obs):
    finite = np.isfinite(phi_obs) & (phi_obs != 0)
    if not np.any(finite):
        return phi_obs, 1.0
    median = np.median(np.abs(phi_obs[finite]))
    if (not np.isfinite(median)) or median == 0:
        return phi_obs, 1.0
    exp = int(np.round(-np.log10(median)))
    scale = 10.0 ** exp
    return phi_obs * scale, scale

phi_true_scaled, phi_scale = scale_phi(fermat_phi_true)

lambda_err = sigma_lambda_frac * np.abs(lambda_true)

# clean
phi_obs_clean_scaled = phi_true_scaled.copy()
t_obs_clean = t_delay_true_mst.copy()
lambda_obs_clean = lambda_true.copy()

# noisy
phi_obs_noisy = fermat_phi_true + rng_np.normal(0.0, sigma_phi_frac * np.abs(fermat_phi_true))
phi_obs_noisy_scaled = phi_obs_noisy * phi_scale

t_obs_noisy = t_delay_true_mst + rng_np.normal(0.0, sigma_t_days, size=len(zl))
lambda_obs_noisy = lambda_true + rng_np.normal(0.0, lambda_err)


def build_data(t_obs, phi_obs_scaled, lambda_obs):
    return {
        "zl": zl,
        "zs": zs,
        "t_obs": t_obs,
        "phi_obs": phi_obs_scaled,
        "phi_scale": phi_scale,
        "lambda_obs": lambda_obs,
        "lambda_err": lambda_err,
    }

sne_data_clean = build_data(t_obs_clean, phi_obs_clean_scaled, lambda_obs_clean)
sne_data_noisy = build_data(t_obs_noisy, phi_obs_noisy_scaled, lambda_obs_noisy)


def cosmology_model(kind, cosmo_prior, sample_h0=True):
    cosmo = {
        "Omegam": numpyro.sample("Omegam", dist.Uniform(cosmo_prior["omegam_low"], cosmo_prior["omegam_up"])),
        "Omegak": 0.0,
        "w0": -1.0,
        "wa": 0.0,
        "h0": 70.0,
    }
    if kind in ["wcdm", "owcdm", "waw0cdm", "owaw0cdm"]:
        cosmo["w0"] = numpyro.sample("w0", dist.Uniform(cosmo_prior["w0_low"], cosmo_prior["w0_up"]))
    if kind in ["waw0cdm", "owaw0cdm"]:
        cosmo["wa"] = numpyro.sample("wa", dist.Uniform(cosmo_prior["wa_low"], cosmo_prior["wa_up"]))
    if kind in ["owcdm", "owaw0cdm"]:
        cosmo["Omegak"] = numpyro.sample("Omegak", dist.Uniform(cosmo_prior["omegak_low"], cosmo_prior["omegak_up"]))
    if sample_h0:
        cosmo["h0"] = numpyro.sample("h0", dist.Uniform(cosmo_prior["h0_low"], cosmo_prior["h0_up"]))
    return cosmo


def sne_model(zl, zs, t_obs, phi_obs, lambda_obs, lambda_err, phi_scale, sigma_t_days=1.0, sigma_phi_frac=0.04):
    cosmo = cosmology_model("waw0cdm", cosmo_prior, sample_h0=True)
    lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.9, 1.1))
    lambda_sigma = numpyro.sample("lambda_sigma", dist.TruncatedNormal(0.05, 0.5, low=0.0, high=0.2))

    zl = jnp.asarray(zl)
    zs = jnp.asarray(zs)
    t_obs = jnp.asarray(t_obs)
    phi_obs = jnp.asarray(phi_obs)
    lambda_obs = jnp.asarray(lambda_obs)
    lambda_err = jnp.asarray(lambda_err)
    phi_scale = jnp.asarray(phi_scale)

    Dl, Ds, Dls = tool.dldsdls(zl, zs, cosmo, n=20)
    Ddt_geom = (1.0 + zl) * Dl * Ds / Dls

    sigma_phi = sigma_phi_frac * phi_obs

    with numpyro.plate("sne", zl.shape[0]):
        phi_true_scaled = numpyro.sample("phi_true_scaled", dist.TruncatedNormal(phi_obs, sigma_phi, low=0.0, high=10.0))
        lambda_true = numpyro.sample("lambda_true", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
        numpyro.sample("lambda_like", dist.Normal(lambda_true, lambda_err), obs=lambda_obs)

        phi_true = phi_true_scaled / phi_scale
        Ddt_true = Ddt_geom * lambda_true
        t_model_days = (Ddt_true * Mpc_km / c_km_day) * phi_true
        numpyro.sample("t_delay_like", dist.Normal(t_model_days, sigma_t_days), obs=t_obs)


def run_mcmc(data, key, tag):
    if TEST_MODE:
        num_warmup, num_samples, num_chains, chain_method = 200, 200, 2, "sequential"
    else:
        num_warmup, num_samples, num_chains, chain_method = 500, 1000, 4, "vectorized"

    nuts = NUTS(sne_model, target_accept_prob=0.85)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(
        key,
        zl=data["zl"],
        zs=data["zs"],
        t_obs=data["t_obs"],
        phi_obs=data["phi_obs"],
        lambda_obs=data["lambda_obs"],
        lambda_err=data["lambda_err"],
        phi_scale=data["phi_scale"],
        sigma_t_days=sigma_t_days,
        sigma_phi_frac=sigma_phi_frac,
    )
    extra = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(extra["diverging"]).sum())
    print(f"[{tag}] divergences: {n_div}")
    posterior = mcmc.get_samples(group_by_chain=True)
    inf_data = az.from_dict(posterior=posterior)
    az.to_netcdf(inf_data, RESULT_DIR / f"sne_{tag}.nc")
    trace_vars = ["h0", "Omegam", "w0", "wa", "lambda_mean", "lambda_sigma"]
    trace_vars = [v for v in trace_vars if v in inf_data.posterior and inf_data.posterior[v].ndim == 2]
    if trace_vars:
        trace_axes = az.plot_trace(inf_data, var_names=trace_vars, compact=False)
        trace_fig = np.asarray(trace_axes).ravel()[0].figure
        trace_fig.savefig(FIG_DIR / f"sne_trace_{tag}.png", dpi=200, bbox_inches="tight")
        plt.close(trace_fig)
    return inf_data


key = random.PRNGKey(42)
key_clean, key_noisy = random.split(key)

idata_clean = run_mcmc(sne_data_clean, key_clean, "clean")
idata_noisy = run_mcmc(sne_data_noisy, key_noisy, "noisy")

corner_vars = select_corner_vars(
    idata_clean,
    idata_noisy,
    ["h0", "Omegam", "w0", "wa", "lambda_mean", "lambda_sigma"],
)
make_overlay_corner(idata_clean, idata_noisy, corner_vars, FIG_DIR / "sne_corner_overlay.png")
