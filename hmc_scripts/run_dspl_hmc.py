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
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, init_to_value
from jax import random
import arviz as az

from slcosmo import tool
from hmc_scripts.corner_utils import select_corner_vars, make_overlay_corner

USE_X64 = os.environ.get("SLCOSMO_USE_X64", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
jax.config.update("jax_enable_x64", USE_X64)
if USE_X64:
    numpyro.enable_x64()
if any(d.platform == "gpu" for d in jax.devices()):
    numpyro.set_platform("gpu")
else:
    numpyro.set_platform("cpu")
print(f"[INFO] Precision mode: {'FP64' if USE_X64 else 'FP32'}", flush=True)

SEED = 42
rng_np = np.random.default_rng(SEED)
np.random.seed(SEED)

TEST_MODE = False
RUN_NOISY_INFERENCE = os.environ.get("SLCOSMO_RUN_NOISY", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
RESULT_DIR = Path("/mnt/lustre/tianli/LensedUniverse_result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("result")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(os.environ.get("SLCOSMO_DATA_DIR", str(workdir / "data")))


def step(message):
    print(f"[STEP] {message}", flush=True)

cosmo_true = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}
cosmo_prior = {
    "w0_up": 0.0,   "w0_low": -2.0,
    "wa_up": 2.0,   "wa_low": -2.0,
    "omegak_up": 1.0, "omegak_low": -1.0,
    "h0_up": 80.0,  "h0_low": 60.0,
    "omegam_up": 0.5, "omegam_low": 0.1,
}
DSPL_TARGET = 500
SIGMA_ETA_LOS_FRAC = 0.01

step("Load DSPL catalog and derive geometric quantities")
data_dspl = np.loadtxt(DATA_DIR / "EuclidDSPLs_1.txt")
data_dspl = data_dspl[(data_dspl[:, 5] < 0.95)]

zl_dspl  = data_dspl[:, 0]
zs1_dspl = data_dspl[:, 1]
zs2_true_cat = data_dspl[:, 2]

beta_err_dspl = data_dspl[:, 6]
model_vel_dspl = data_dspl[:, 11]

step(
    f"Catalog zs2<=zs1 count: {int(np.sum(zs2_true_cat <= zs1_dspl))}/"
    f"{len(zs2_true_cat)} (no pre-filter)"
)

N_all = len(zl_dspl)
if N_all > DSPL_TARGET:
    select_idx = np.sort(rng_np.choice(N_all, size=DSPL_TARGET, replace=False))
    zl_dspl = zl_dspl[select_idx]
    zs1_dspl = zs1_dspl[select_idx]
    zs2_true_cat = zs2_true_cat[select_idx]
    beta_err_dspl = beta_err_dspl[select_idx]
    model_vel_dspl = model_vel_dspl[select_idx]
N_dspl = len(zl_dspl)

step(f"Use {N_dspl} DSPL systems; all source2 redshifts are treated as precise")

Dl1, Ds1, Dls1 = tool.compute_distances(zl_dspl, zs1_dspl, cosmo_true)
Dl2, Ds2, Dls2 = tool.compute_distances(zl_dspl, zs2_true_cat, cosmo_true)
beta_geom_dspl = Dls1 * Ds2 / (Ds1 * Dls2)

step("Build clean/noisy mock DSPL observables")
lambda_true = tool.truncated_normal(1.0, 0.05, 0.85, 1.15, N_dspl, random_state=rng_np)
lambda_err = lambda_true * 0.10

true_vel = model_vel_dspl * jnp.sqrt(lambda_true)
vel_err = 0.03 * true_vel

beta_true = tool.beta_antimst(beta_geom_dspl, mst=lambda_true)

lambda_obs_clean = lambda_true
beta_obs_clean = beta_true

lambda_obs_noisy = lambda_true + rng_np.normal(0.0, lambda_err)
beta_err_tot_dspl = np.sqrt(beta_err_dspl**2 + (SIGMA_ETA_LOS_FRAC * np.asarray(beta_true))**2)
beta_obs_noisy = tool.truncated_normal(beta_true, beta_err_tot_dspl, 0.0, 1.0, random_state=rng_np)


def build_data(lambda_obs, beta_obs):
    return {
        "zl": zl_dspl,
        "zs1": zs1_dspl,
        "zs2_cat": zs2_true_cat,
        "zs2_obs": zs2_true_cat,
        "beta_obs": beta_obs,
        "beta_err": beta_err_dspl,
        "v_model": model_vel_dspl,
        "v_obs": true_vel,
        "v_err": vel_err,
        "lambda_err": lambda_err,
        "lambda_obs": lambda_obs,
    }


dspl_data_clean = build_data(lambda_obs_clean, beta_obs_clean)
dspl_data_noisy = build_data(lambda_obs_noisy, beta_obs_noisy)


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
    zs2 = jnp.asarray(dspl_data["zs2_obs"])

    Dl1, Ds1, Dls1 = tool.compute_distances(zl, zs1, cosmo)
    Dl2, Ds2, Dls2 = tool.compute_distances(zl, zs2, cosmo)
    beta_geom = Dls1 * Ds2 / (Ds1 * Dls2)

    with numpyro.plate("dspl", len(zl)):
        lambda_dspl = numpyro.sample("lambda_dspl", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.5, high=1.5))
        numpyro.sample("lambda_dspl_like", dist.Normal(lambda_dspl, dspl_data["lambda_err"]), obs=dspl_data["lambda_obs"])
        beta_mst = tool.beta_antimst(beta_geom, lambda_dspl)
        beta_err_tot = jnp.sqrt(jnp.asarray(dspl_data["beta_err"])**2 + (SIGMA_ETA_LOS_FRAC * beta_mst)**2)
        numpyro.sample(
            "beta_dspl_like",
            dist.TruncatedNormal(beta_mst, beta_err_tot, low=0.0, high=1.0),
            obs=dspl_data["beta_obs"],
        )


def build_init_values(dspl_data):
    init_values = {
        "Omegam": jnp.asarray(cosmo_true["Omegam"]),
        "w0": jnp.asarray(cosmo_true["w0"]),
        "wa": jnp.asarray(cosmo_true["wa"]),
        "h0": jnp.asarray(cosmo_true["h0"]),
        "lambda_mean": jnp.asarray(1.0),
        "lambda_sigma": jnp.asarray(0.08),
    }
    lambda_dspl = np.asarray(dspl_data["lambda_obs"], dtype=np.float64)
    lambda_dspl = np.clip(lambda_dspl, 0.801, 1.199)
    init_values["lambda_dspl"] = jnp.asarray(lambda_dspl)
    return init_values


def run_mcmc(data, key, tag):
    step(f"Run MCMC for DSPL ({tag})")
    if TEST_MODE:
        num_warmup, num_samples, num_chains, chain_method = 200, 200, 2, "sequential"
    else:
        num_warmup, num_samples, num_chains, chain_method = 500, 1500, 4, "vectorized"

    nuts = NUTS(
        dspl_model,
        target_accept_prob=0.9,
        init_strategy=init_to_value(values=build_init_values(data)),
    )
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(key, dspl_data=data)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(extra["diverging"]).sum())
    print(f"[{tag}] divergences: {n_div}")
    posterior = mcmc.get_samples(group_by_chain=True)
    inf_data = az.from_dict(posterior=posterior)
    az.to_netcdf(inf_data, RESULT_DIR / f"dspl_{tag}.nc")
    trace_vars = ["h0", "Omegam", "w0", "wa", "lambda_mean", "lambda_sigma"]
    trace_vars = [v for v in trace_vars if v in inf_data.posterior and inf_data.posterior[v].ndim == 2]
    if trace_vars:
        trace_axes = az.plot_trace(inf_data, var_names=trace_vars, compact=False)
        trace_fig = np.asarray(trace_axes).ravel()[0].figure
        trace_fig.savefig(FIG_DIR / f"dspl_trace_{tag}.pdf", dpi=200, bbox_inches="tight")
        plt.close(trace_fig)
    return inf_data


key = random.PRNGKey(42)
if RUN_NOISY_INFERENCE:
    key_clean, key_noisy = random.split(key)
else:
    key_clean = key

step("Execute clean run")
idata_clean = run_mcmc(dspl_data_clean, key_clean, "clean")

if RUN_NOISY_INFERENCE:
    step("Execute noisy run")
    idata_noisy = run_mcmc(dspl_data_noisy, key_noisy, "noisy")

    step("Create overlay corner plot")
    corner_vars = select_corner_vars(
        idata_clean,
        idata_noisy,
        ["h0", "Omegam", "w0", "wa", "lambda_mean", "lambda_sigma"],
    )
    make_overlay_corner(idata_clean, idata_noisy, corner_vars, FIG_DIR / "dspl_corner_overlay.pdf")
else:
    step("Skip noisy inference (RUN_NOISY_INFERENCE=False)")
