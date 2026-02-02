#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import sys

workdir = Path(__file__).resolve().parent.parent
os.chdir(workdir)
sys.path.insert(0, str(workdir))

import json
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from jax import random
import arviz as az

from slcosmo.tools import tool

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

cosmo_true = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}
cosmo_prior = {
    "w0_up": 0.0,   "w0_low": -2.0,
    "wa_up": 2.0,   "wa_low": -2.0,
    "omegak_up": 1.0, "omegak_low": -1.0,
    "h0_up": 80.0,  "h0_low": 60.0,
    "omegam_up": 0.5, "omegam_low": 0.1,
}

DATA_JSON = Path("../Temp_data/static_datavectors_seed6.json")
with DATA_JSON.open("r") as f:
    blocks = json.load(f)

z_lens_list = []
z_src_list = []
t_base_list = []
t_err_list = []
block_id_list = []

for b, block in enumerate(blocks):
    z_lens = np.asarray(block["z_lens"], dtype=float)
    z_src = np.asarray(block["z_src"], dtype=float)

    td = np.asarray(block["td_measured"], dtype=float)
    td_mean = td.mean(axis=1)

    n_lens, n_td = td_mean.shape
    if n_td == 3:
        idx = np.argmax(np.abs(td_mean), axis=1)
        t_base = np.abs(td_mean[np.arange(n_lens), idx])
    else:
        t_base = np.abs(td_mean[:, 0])

    if b in (0, 1, 2):
        t_err = 0.03 * t_base
    else:
        t_err = np.full_like(t_base, 5.0)

    z_lens_list.append(z_lens)
    z_src_list.append(z_src)
    t_base_list.append(t_base)
    t_err_list.append(t_err)
    block_id_list.append(np.full(n_lens, b, dtype=int))

z_lens = np.concatenate(z_lens_list)
z_src = np.concatenate(z_src_list)
t_base = np.concatenate(t_base_list)
t_err = np.concatenate(t_err_list)
block_id = np.concatenate(block_id_list)

zl_j = jnp.asarray(z_lens)
zs_j = jnp.asarray(z_src)
Dl, Ds, Dls = tool.dldsdls(zl_j, zs_j, cosmo_true, n=20)
Ddt_geom = (1.0 + zl_j) * Dl * Ds / Dls
Ddt_geom = np.asarray(Ddt_geom)

c_km_day = tool.c_km_s * 86400.0
Mpc_km = tool.Mpc / 1000.0

phi_true = (c_km_day * t_base) / (Ddt_geom * Mpc_km)

phi_err_frac_by_block = {
    0: 0.02,
    1: 0.05,
    2: 0.05,
    3: 0.11,
    4: 0.11,
    5: 0.18,
    6: 0.18,
    7: 0.18,
    8: 0.18,
}

sigma_v_frac_by_block = {
    0: 0.03,
    1: 0.03,
    2: 0.03,
    3: 0.10,
    4: 0.10,
    5: 0.10,
    6: 0.10,
    7: np.nan,
    8: np.nan,
}

phi_err_frac = np.asarray([phi_err_frac_by_block[b] for b in block_id])
phi_err = phi_err_frac * np.abs(phi_true)

lambda_pop_mean = 1.0
lambda_pop_sigma = 0.05
lambda_low, lambda_high = 0.8, 1.2
lambda_true = tool.truncated_normal(lambda_pop_mean, lambda_pop_sigma, lambda_low, lambda_high, z_lens.size, random_state=rng_np)

sigma_v_frac = np.asarray([sigma_v_frac_by_block[b] for b in block_id])
mst_mask = np.isfinite(sigma_v_frac)
mst_err_frac = 2.0 * sigma_v_frac
lambda_err = np.where(mst_mask, mst_err_frac * np.abs(lambda_true), lambda_pop_sigma)

t_true = t_base * lambda_true

def scale_phi(phi_in):
    finite = np.isfinite(phi_in) & (phi_in != 0)
    if not np.any(finite):
        return phi_in, 1.0
    median = np.median(np.abs(phi_in[finite]))
    if (not np.isfinite(median)) or median == 0:
        return phi_in, 1.0
    exp = int(np.round(-np.log10(median)))
    scale = 10.0 ** exp
    return phi_in * scale, scale

phi_true_scaled, phi_scale = scale_phi(phi_true)

phi_obs_clean = phi_true.copy()
phi_obs_noisy = phi_true + rng_np.normal(0.0, phi_err)

phi_obs_clean_scaled = phi_obs_clean * phi_scale
phi_obs_noisy_scaled = phi_obs_noisy * phi_scale

lambda_obs_clean = lambda_true.copy()
lambda_obs_noisy = lambda_true + rng_np.normal(0.0, lambda_err)

t_obs_clean = t_true.copy()
t_obs_noisy = t_true + rng_np.normal(0.0, t_err)


def build_data(t_obs, phi_obs_scaled, lambda_obs):
    return {
        "zl": z_lens,
        "zs": z_src,
        "t_obs": t_obs,
        "t_err": t_err,
        "phi_obs": phi_obs_scaled,
        "phi_err": phi_err_frac * np.abs(phi_obs_scaled),
        "phi_scale": phi_scale,
        "lambda_obs": lambda_obs,
        "lambda_err": lambda_err,
        "mst_mask": mst_mask,
    }

quasar_data_clean = build_data(t_obs_clean, phi_obs_clean_scaled, lambda_obs_clean)
quasar_data_noisy = build_data(t_obs_noisy, phi_obs_noisy_scaled, lambda_obs_noisy)


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


def quasar_model(zl, zs, t_obs, t_err, phi_obs, phi_err, phi_scale, lambda_obs, lambda_err, mst_mask):
    cosmo = cosmology_model("waw0cdm", cosmo_prior, sample_h0=True)

    zl = jnp.asarray(zl)
    zs = jnp.asarray(zs)
    t_obs = jnp.asarray(t_obs)
    t_err = jnp.asarray(t_err)
    phi_obs = jnp.asarray(phi_obs)
    phi_err = jnp.asarray(phi_err)
    phi_scale = jnp.asarray(phi_scale)
    lambda_obs = jnp.asarray(lambda_obs)
    lambda_err = jnp.asarray(lambda_err)
    mst_mask = jnp.asarray(mst_mask)

    lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.9, 1.1))
    lambda_sigma = numpyro.sample("lambda_sigma", dist.TruncatedNormal(0.05, 0.5, low=0.0, high=0.2))
    with numpyro.plate("lens", zl.shape[0]):
        lambda_true = numpyro.sample("lambda_true", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
        numpyro.sample("lambda_like", dist.Normal(lambda_true, lambda_err).mask(mst_mask), obs=lambda_obs)

    Dl, Ds, Dls = tool.dldsdls(zl, zs, cosmo, n=20)
    Ddt_geom = (1.0 + zl) * Dl * Ds / Dls

    with numpyro.plate("td_obs", zl.shape[0]):
        phi_true_scaled = numpyro.sample("phi_true_scaled", dist.Normal(phi_obs, phi_err))
        phi_true = phi_true_scaled / phi_scale
        t_model_days = (Ddt_geom * Mpc_km / c_km_day) * phi_true
        t_model_days = t_model_days * lambda_true
        numpyro.sample("t_delay_like", dist.Normal(t_model_days, t_err), obs=t_obs)


def run_mcmc(data, key, tag):
    if TEST_MODE:
        num_warmup, num_samples, num_chains, chain_method = 200, 200, 2, "sequential"
    else:
        num_warmup, num_samples, num_chains, chain_method = 500, 1000, 4, "vectorized"

    nuts = NUTS(quasar_model, target_accept_prob=0.85)
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
        t_err=data["t_err"],
        phi_obs=data["phi_obs"],
        phi_err=data["phi_err"],
        phi_scale=data["phi_scale"],
        lambda_obs=data["lambda_obs"],
        lambda_err=data["lambda_err"],
        mst_mask=data["mst_mask"],
    )
    posterior = mcmc.get_samples(group_by_chain=True)
    inf_data = az.from_dict(posterior=posterior)
    az.to_netcdf(inf_data, RESULT_DIR / f"quasar_{tag}.nc")


key = random.PRNGKey(42)
key_clean, key_noisy = random.split(key)

run_mcmc(quasar_data_clean, key_clean, "clean")
run_mcmc(quasar_data_noisy, key_noisy, "noisy")
