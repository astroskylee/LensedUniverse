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

LUT = np.load(DATA_DIR / "velocity_disp_table.npy")
N1, N2, N3, N4 = LUT.shape
thetaE_grid = np.linspace(0.5, 3.0, N1)
gamma_grid  = np.linspace(1.2, 2.8, N2)
Re_grid     = np.linspace(0.15, 3.0, N3)
beta_grid   = np.linspace(-0.5, 0.8, N4)
jampy_interp = tool.make_4d_interpolant(thetaE_grid, gamma_grid, Re_grid, beta_grid, LUT)

Euclid_GG_data = np.loadtxt(DATA_DIR / "Euclid_len.txt")
zl_lens = Euclid_GG_data[:, 0]
zs_lens = Euclid_GG_data[:, 1]
Ein_lens = Euclid_GG_data[:, 2]
re_lens = Euclid_GG_data[:, 5]

mask_lens = (Ein_lens >= 0.6) & (re_lens >= 0.25) & (re_lens <= 2.8)
zl_lens = zl_lens[mask_lens]
zs_lens = zs_lens[mask_lens]
thetaE_lens = Ein_lens[mask_lens]
re_lens = re_lens[mask_lens]

_dl, ds_lens, dls_lens = tool.dldsdls(zl_lens, zs_lens, cosmo_true, n=20)
N_lens = len(zl_lens)

gamma_true = tool.truncated_normal(2.0, 0.2, 1.5, 2.5, N_lens, random_state=rng_np)
beta_true = tool.truncated_normal(0.0, 0.2, -0.4, 0.4, N_lens, random_state=rng_np)
vel_model = jampy_interp(thetaE_lens, gamma_true, re_lens, beta_true) * jnp.sqrt(ds_lens / dls_lens)

lambda_true = tool.truncated_normal(1.0, 0.05, 0.8, 1.2, N_lens, random_state=rng_np)
vel_true = vel_model * jnp.sqrt(lambda_true)

theta_E_err = 0.01 * thetaE_lens
vel_err = 0.10 * vel_true

gamma_obs_clean = gamma_true
theta_E_obs_clean = thetaE_lens
vel_obs_clean = vel_true

gamma_obs_noisy = gamma_true + tool.truncated_normal(0.0, 0.05, -0.2, 0.2, N_lens, random_state=rng_np)
theta_E_obs_noisy = thetaE_lens + np.random.normal(0.0, theta_E_err)
vel_obs_noisy = np.random.normal(vel_true, vel_err)

def build_data(gamma_obs, theta_E_obs, vel_obs):
    return {
        "zl": zl_lens,
        "zs": zs_lens,
        "theta_E": theta_E_obs,
        "theta_E_err": theta_E_err,
        "re": re_lens,
        "gamma_obs": gamma_obs,
        "vel_obs": vel_obs,
        "vel_err": vel_err,
    }

lens_data_clean = build_data(gamma_obs_clean, theta_E_obs_clean, vel_obs_clean)
lens_data_noisy = build_data(gamma_obs_noisy, theta_E_obs_noisy, vel_obs_noisy)

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


def lens_model(lens_data):
    cosmo = cosmology_model("waw0cdm", cosmo_prior, sample_h0=True)

    lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.9, 1.1))
    lambda_sigma = numpyro.sample("lambda_sigma", dist.TruncatedNormal(0.05, 0.5, low=0.0, high=0.2))

    gamma_mean = numpyro.sample("gamma_mean", dist.Uniform(1.6, 2.4))
    gamma_sigma = numpyro.sample("gamma_sigma", dist.TruncatedNormal(0.2, 0.2, low=0.0, high=0.4))

    beta_mean = numpyro.sample("beta_mean", dist.Uniform(-0.3, 0.3))
    beta_sigma = numpyro.sample("beta_sigma", dist.TruncatedNormal(0.2, 0.2, low=0.0, high=0.4))

    dl_lens, ds_lens, dls_lens = tool.dldsdls(lens_data["zl"], lens_data["zs"], cosmo, n=20)
    N = len(lens_data["zl"])
    with numpyro.plate("lens", N):
        gamma_i = numpyro.sample("gamma_i", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.6, high=2.4))
        beta_i = numpyro.sample("beta_i", dist.TruncatedNormal(beta_mean, beta_sigma, low=-0.4, high=0.4))
        lambda_lens = numpyro.sample("lambda_lens", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
        theta_E_i = numpyro.sample("theta_E_i", dist.Normal(lens_data["theta_E"], lens_data["theta_E_err"]))
        v_interp = jampy_interp(theta_E_i, gamma_i, lens_data["re"], beta_i)
        vel_pred = v_interp * jnp.sqrt(ds_lens / dls_lens) * jnp.sqrt(lambda_lens)

        numpyro.sample("gamma_obs_lens", dist.Normal(gamma_i, 0.05), obs=lens_data["gamma_obs"])
        numpyro.sample("vel_lens_like", dist.Normal(vel_pred, lens_data["vel_err"]), obs=lens_data["vel_obs"])


def run_mcmc(data, key, tag):
    if TEST_MODE:
        num_warmup, num_samples, num_chains, chain_method = 200, 200, 2, "sequential"
    else:
        num_warmup, num_samples, num_chains, chain_method = 500, 1500, 4, "vectorized"

    nuts = NUTS(lens_model, target_accept_prob=0.85)
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(key, lens_data=data)
    posterior = mcmc.get_samples(group_by_chain=True)
    inf_data = az.from_dict(posterior=posterior)
    az.to_netcdf(inf_data, RESULT_DIR / f"lens_kin_{tag}.nc")


key = random.PRNGKey(42)
key_clean, key_noisy = random.split(key)

run_mcmc(lens_data_clean, key_clean, "clean")
run_mcmc(lens_data_noisy, key_noisy, "noisy")
