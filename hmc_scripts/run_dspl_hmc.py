#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import sys

workdir = Path(__file__).resolve().parent.parent
os.chdir(workdir)
sys.path.insert(0, str(workdir))

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from jax import random
import arviz as az

from slcosmo import tool

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

DATA_DIR = Path(os.environ.get("SLCOSMO_DATA_DIR", str(workdir / "data")))

cosmo_true = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}
cosmo_prior = {
    "w0_up": 0.0,   "w0_low": -2.0,
    "wa_up": 2.0,   "wa_low": -2.0,
    "omegak_up": 1.0, "omegak_low": -1.0,
    "h0_up": 80.0,  "h0_low": 60.0,
    "omegam_up": 0.5, "omegam_low": 0.1,
}

data_dspl = np.loadtxt(DATA_DIR / "EuclidDSPLs_1.txt")
data_dspl = data_dspl[(data_dspl[:, 5] < 0.95)]

zl_dspl  = data_dspl[:, 0]
zs1_dspl = data_dspl[:, 1]
zs2_true_cat = data_dspl[:, 2]

beta_err_dspl = data_dspl[:, 6]
model_vel_dspl = data_dspl[:, 11]

m_ok = (zs2_true_cat > zs1_dspl)
zl_dspl  = zl_dspl[m_ok]
zs1_dspl = zs1_dspl[m_ok]
zs2_true_cat = zs2_true_cat[m_ok]
beta_err_dspl = beta_err_dspl[m_ok]
model_vel_dspl = model_vel_dspl[m_ok]

N_dspl = len(zl_dspl)

is_photo = (rng_np.random(N_dspl) < 0.60)
zs2_err = np.where(is_photo, 0.1, 1e-4)
zs2_obs = zs2_true_cat + rng_np.normal(0.0, zs2_err)

eps = 1e-3
bad = zs2_obs <= (zs1_dspl + eps)
for _ in range(20):
    if not np.any(bad):
        break
    zs2_obs[bad] = zs2_true_cat[bad] + rng_np.normal(0.0, zs2_err[bad])
    bad = zs2_obs <= (zs1_dspl + eps)
zs2_obs = np.maximum(zs2_obs, zs1_dspl + eps)

Dl1, Ds1, Dls1 = tool.compute_distances(zl_dspl, zs1_dspl, cosmo_true)
Dl2, Ds2, Dls2 = tool.compute_distances(zl_dspl, zs2_true_cat, cosmo_true)
beta_geom_dspl = Dls1 * Ds2 / (Ds1 * Dls2)

lambda_true = tool.truncated_normal(1.0, 0.05, 0.85, 1.15, N_dspl, random_state=rng_np)
lambda_err = lambda_true * 0.06

true_vel = model_vel_dspl * jnp.sqrt(lambda_true)
vel_err = 0.03 * true_vel

beta_true = tool.beta_antimst(beta_geom_dspl, mst=lambda_true)

lambda_obs_clean = lambda_true
beta_obs_clean = beta_true
zs2_use_clean = zs2_true_cat

lambda_obs_noisy = lambda_true + np.random.normal(0.0, lambda_err)
beta_obs_noisy = tool.truncated_normal(beta_true, beta_err_dspl, 0.0, 1.0, random_state=rng_np)
zs2_use_noisy = zs2_obs


def build_data(lambda_obs, beta_obs, zs2_use):
    return {
        "zl": zl_dspl,
        "zs1": zs1_dspl,
        "zs2_cat": zs2_true_cat,
        "zs2_obs": zs2_use,
        "zs2_err": zs2_err,
        "is_photo": is_photo.astype(np.int32),
        "beta_obs": beta_obs,
        "beta_err": beta_err_dspl,
        "v_model": model_vel_dspl,
        "v_obs": true_vel,
        "v_err": vel_err,
        "lambda_err": lambda_err,
        "lambda_obs": lambda_obs,
    }


dspl_data_clean = build_data(lambda_obs_clean, beta_obs_clean, zs2_use_clean)
dspl_data_noisy = build_data(lambda_obs_noisy, beta_obs_noisy, zs2_use_noisy)


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


def dspl_model(dspl_data):
    cosmo = cosmology_model("waw0cdm", cosmo_prior, sample_h0=True)

    lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.9, 1.1))
    lambda_sigma = numpyro.sample("lambda_sigma", dist.TruncatedNormal(0.05, 0.5, low=0.0, high=0.2))

    zl = jnp.asarray(dspl_data["zl"])
    zs1 = jnp.asarray(dspl_data["zs1"])
    zs2_obs = jnp.asarray(dspl_data["zs2_obs"])
    zs2_err = jnp.asarray(dspl_data["zs2_err"])

    Dl1, Ds1, Dls1 = tool.compute_distances(zl, zs1, cosmo)

    eps = 1e-3
    zs2_true = numpyro.sample(
        "zs2_true",
        dist.TruncatedNormal(zs2_obs, zs2_err, low=zs1 + eps, high=10.0).to_event(1),
    )

    Dl2, Ds2, Dls2 = tool.compute_distances(zl, zs2_true, cosmo)
    beta_geom = Dls1 * Ds2 / (Ds1 * Dls2)

    N = len(zl)
    with numpyro.plate("dspl", N):
        lambda_dspl = numpyro.sample("lambda_dspl", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
        numpyro.sample("lambda_dspl_like", dist.Normal(lambda_dspl, dspl_data["lambda_err"]), obs=dspl_data["lambda_obs"])
        beta_mst = tool.beta_antimst(beta_geom, lambda_dspl)
        numpyro.sample("beta_dspl_like", dist.TruncatedNormal(beta_mst, dspl_data["beta_err"], low=0.0, high=1.0), obs=dspl_data["beta_obs"])


def run_mcmc(data, key, tag):
    if TEST_MODE:
        num_warmup, num_samples, num_chains, chain_method = 200, 200, 2, "sequential"
    else:
        num_warmup, num_samples, num_chains, chain_method = 500, 1500, 4, "vectorized"

    nuts = NUTS(dspl_model, target_accept_prob=0.85)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(key, dspl_data=data)
    posterior = mcmc.get_samples(group_by_chain=True)
    inf_data = az.from_dict(posterior=posterior)
    az.to_netcdf(inf_data, RESULT_DIR / f"dspl_{tag}.nc")


key = random.PRNGKey(42)
key_clean, key_noisy = random.split(key)

run_mcmc(dspl_data_clean, key_clean, "clean")
run_mcmc(dspl_data_noisy, key_noisy, "noisy")
