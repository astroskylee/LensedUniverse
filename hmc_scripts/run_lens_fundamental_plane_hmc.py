#!/usr/bin/env python3
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
import sys

workdir = Path(__file__).resolve().parent.parent
os.chdir(workdir)
sys.path.insert(0, str(workdir))

import numpy as np
import pandas as pd
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

USE_X64 = os.environ.get("SLCOSMO_USE_X64", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
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
RESULT_DIR = Path("/mnt/lustre/tianli/LensedUniverse_result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("result")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(os.environ.get("SLCOSMO_DATA_DIR", str(workdir / "data")))
DATA_CSV = DATA_DIR / "db_zBEAMS_PEMD_100000_s1_GDB_phot_err_ManySF_TL.csv"


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
Z_SEP_MIN = 0.2
VEL_FRAC_ERR_MAX = 0.30

step("Load lookup table and fundamental-plane catalog")
LUT = np.load(DATA_DIR / "velocity_disp_table.npy")
N1, N2, N3, N4 = LUT.shape
thetaE_grid = np.linspace(0.5, 3.0, N1)
gamma_grid = np.linspace(1.2, 2.8, N2)
Re_grid = np.linspace(0.15, 3.0, N3)
beta_grid = np.linspace(-0.5, 0.8, N4)
jampy_interp = tool.make_4d_interpolant(thetaE_grid, gamma_grid, Re_grid, beta_grid, LUT)

fp_df = pd.read_csv(DATA_CSV)
if TEST_MODE:
    n_use = 200
else:
    n_use = 100000
select_idx = rng_np.choice(fp_df.shape[0], size=min(n_use, fp_df.shape[0]), replace=False)
fp_df = fp_df.iloc[np.sort(select_idx)].reset_index(drop=True)
step(f"Sampled {fp_df.shape[0]} FP systems before observation cuts")

zl_fp = fp_df["zL_true"].to_numpy()
zs_fp = fp_df["zS_true"].to_numpy()
zL_sigma_fp = np.abs(fp_df["sigma_zL_obs"].to_numpy())
zS_sigma_fp = np.abs(fp_df["sigma_zS_obs"].to_numpy())
thetaE_true_fp = fp_df["tE_true"].to_numpy()
veldisp_true_catalog = fp_df["veldisp_true"].to_numpy()
vel_frac_err_template = fp_df["sigma_veldisp_obs"].to_numpy() / veldisp_true_catalog

# Build Re_true directly from veldisp_true by linear compression to [0.5, 2.8] arcsec.
vel_min = np.min(veldisp_true_catalog)
vel_max = np.max(veldisp_true_catalog)
re_fp = 0.5 + (veldisp_true_catalog - vel_min) * (2.8 - 0.5) / (vel_max - vel_min)
step(
    "Re_true from linear veldisp mapping: "
    f"vmin={vel_min:.3f}, vmax={vel_max:.3f}, "
    f"re_min={re_fp.min():.4f}, re_median={np.median(re_fp):.4f}, re_max={re_fp.max():.4f}"
)

theta_keep_mask = thetaE_true_fp <= 2.5
step(f"Keep {int(theta_keep_mask.sum())}/{theta_keep_mask.size} FP systems after thetaE_true<=2.5 cut")
zl_fp = zl_fp[theta_keep_mask]
zs_fp = zs_fp[theta_keep_mask]
zL_sigma_fp = zL_sigma_fp[theta_keep_mask]
zS_sigma_fp = zS_sigma_fp[theta_keep_mask]
thetaE_true_fp = thetaE_true_fp[theta_keep_mask]
re_fp = re_fp[theta_keep_mask]
veldisp_true_catalog = veldisp_true_catalog[theta_keep_mask]
vel_frac_err_template = vel_frac_err_template[theta_keep_mask]
thetaE_err_fp = 0.01 * thetaE_true_fp

vel_err_keep_mask = vel_frac_err_template < VEL_FRAC_ERR_MAX
step(
    f"Keep {int(vel_err_keep_mask.sum())}/{vel_err_keep_mask.size} FP systems after "
    f"frac vel err<{VEL_FRAC_ERR_MAX:.2f} cut"
)
zl_fp = zl_fp[vel_err_keep_mask]
zs_fp = zs_fp[vel_err_keep_mask]
zL_sigma_fp = zL_sigma_fp[vel_err_keep_mask]
zS_sigma_fp = zS_sigma_fp[vel_err_keep_mask]
thetaE_true_fp = thetaE_true_fp[vel_err_keep_mask]
thetaE_err_fp = thetaE_err_fp[vel_err_keep_mask]
re_fp = re_fp[vel_err_keep_mask]
veldisp_true_catalog = veldisp_true_catalog[vel_err_keep_mask]
vel_frac_err_template = vel_frac_err_template[vel_err_keep_mask]

_dl_fp_true, ds_fp_true, dls_fp_true = tool.dldsdls(zl_fp, zs_fp, cosmo_true, n=20)
step("Build clean/noisy mock fundamental-plane observables")
gamma_true_fp = tool.truncated_normal(2.0, 0.2, 1.5, 2.5, zl_fp.size, random_state=rng_np)
beta_true_fp = tool.truncated_normal(0.0, 0.2, -0.4, 0.4, zl_fp.size, random_state=rng_np)
lambda_true_fp = tool.truncated_normal(1.0, 0.05, 0.8, 1.2, zl_fp.size, random_state=rng_np)
vel_model_fp = jampy_interp(thetaE_true_fp, gamma_true_fp, re_fp, beta_true_fp) * jnp.sqrt(ds_fp_true / dls_fp_true)
vel_true_fp = np.asarray(vel_model_fp) * np.sqrt(lambda_true_fp)
vel_err_fp = vel_frac_err_template * np.abs(vel_true_fp)

clean_keep_mask = vel_true_fp <= 400.0
step(f"Keep {int(clean_keep_mask.sum())}/{clean_keep_mask.size} FP systems after clean vel<=400 cut")
zl_fp = zl_fp[clean_keep_mask]
zs_fp = zs_fp[clean_keep_mask]
zL_sigma_fp = zL_sigma_fp[clean_keep_mask]
zS_sigma_fp = zS_sigma_fp[clean_keep_mask]
thetaE_true_fp = thetaE_true_fp[clean_keep_mask]
thetaE_err_fp = thetaE_err_fp[clean_keep_mask]
re_fp = re_fp[clean_keep_mask]
gamma_true_fp = gamma_true_fp[clean_keep_mask]
beta_true_fp = beta_true_fp[clean_keep_mask]
lambda_true_fp = lambda_true_fp[clean_keep_mask]
vel_true_fp = vel_true_fp[clean_keep_mask]
vel_err_fp = vel_err_fp[clean_keep_mask]

thetaE_obs_clean_fp = thetaE_true_fp.copy()
vel_obs_clean_fp = vel_true_fp.copy()
thetaE_obs_noisy_fp = thetaE_true_fp + rng_np.normal(0.0, thetaE_err_fp)
vel_obs_noisy_fp = tool.truncated_normal(vel_true_fp, vel_err_fp, 100.0, 500.0, random_state=rng_np)
vel_q = np.quantile(vel_obs_clean_fp, [0.01, 0.05, 0.50, 0.95, 0.99])
step(
    "vel_obs_clean_fp before noisy cuts: "
    f"min={vel_obs_clean_fp.min():.3f}, q01={vel_q[0]:.3f}, q05={vel_q[1]:.3f}, "
    f"median={vel_q[2]:.3f}, q95={vel_q[3]:.3f}, q99={vel_q[4]:.3f}, max={vel_obs_clean_fp.max():.3f}"
)

noisy_keep_mask = vel_obs_noisy_fp <= 400.0
step(f"Keep {int(noisy_keep_mask.sum())}/{noisy_keep_mask.size} FP systems after noisy vel<=400 cut")

zl_fp = zl_fp[noisy_keep_mask]
zs_fp = zs_fp[noisy_keep_mask]
zL_sigma_fp = zL_sigma_fp[noisy_keep_mask]
zS_sigma_fp = zS_sigma_fp[noisy_keep_mask]
thetaE_true_fp = thetaE_true_fp[noisy_keep_mask]
thetaE_err_fp = thetaE_err_fp[noisy_keep_mask]
re_fp = re_fp[noisy_keep_mask]
gamma_true_fp = gamma_true_fp[noisy_keep_mask]
beta_true_fp = beta_true_fp[noisy_keep_mask]
lambda_true_fp = lambda_true_fp[noisy_keep_mask]
vel_err_fp = vel_err_fp[noisy_keep_mask]
thetaE_obs_clean_fp = thetaE_obs_clean_fp[noisy_keep_mask]
vel_obs_clean_fp = vel_obs_clean_fp[noisy_keep_mask]
thetaE_obs_noisy_fp = thetaE_obs_noisy_fp[noisy_keep_mask]
vel_obs_noisy_fp = vel_obs_noisy_fp[noisy_keep_mask]

z_true_mask = (zl_fp > 0.0) & (zs_fp > zl_fp + Z_SEP_MIN)
step(
    f"Keep {int(z_true_mask.sum())}/{z_true_mask.size} FP systems after true-z cut "
    f"(zs-zl>{Z_SEP_MIN:.1f})"
)
zl_fp = zl_fp[z_true_mask]
zs_fp = zs_fp[z_true_mask]
zL_sigma_fp = zL_sigma_fp[z_true_mask]
zS_sigma_fp = zS_sigma_fp[z_true_mask]
thetaE_true_fp = thetaE_true_fp[z_true_mask]
thetaE_err_fp = thetaE_err_fp[z_true_mask]
re_fp = re_fp[z_true_mask]
gamma_true_fp = gamma_true_fp[z_true_mask]
beta_true_fp = beta_true_fp[z_true_mask]
lambda_true_fp = lambda_true_fp[z_true_mask]
vel_err_fp = vel_err_fp[z_true_mask]
thetaE_obs_clean_fp = thetaE_obs_clean_fp[z_true_mask]
vel_obs_clean_fp = vel_obs_clean_fp[z_true_mask]
thetaE_obs_noisy_fp = thetaE_obs_noisy_fp[z_true_mask]
vel_obs_noisy_fp = vel_obs_noisy_fp[z_true_mask]

zl_obs_clean_fp = zl_fp.copy()
zs_obs_clean_fp = zs_fp.copy()
zl_obs_noisy_fp = zl_fp + rng_np.normal(0.0, zL_sigma_fp)
zs_obs_noisy_fp = zs_fp + rng_np.normal(0.0, zS_sigma_fp)
bad_z_mask = (
    (zl_obs_noisy_fp <= 0.0)
    | (zs_obs_noisy_fp <= 0.0)
    | (zs_obs_noisy_fp <= zl_obs_noisy_fp + Z_SEP_MIN)
)
for _ in range(20):
    if not np.any(bad_z_mask):
        break
    zl_obs_noisy_fp[bad_z_mask] = zl_fp[bad_z_mask] + rng_np.normal(0.0, zL_sigma_fp[bad_z_mask])
    zs_obs_noisy_fp[bad_z_mask] = zs_fp[bad_z_mask] + rng_np.normal(0.0, zS_sigma_fp[bad_z_mask])
    bad_z_mask = (
        (zl_obs_noisy_fp <= 0.0)
        | (zs_obs_noisy_fp <= 0.0)
        | (zs_obs_noisy_fp <= zl_obs_noisy_fp + Z_SEP_MIN)
    )
zl_obs_noisy_fp = np.maximum(zl_obs_noisy_fp, 1e-4)
zs_obs_noisy_fp = np.maximum(zs_obs_noisy_fp, zl_obs_noisy_fp + Z_SEP_MIN + 1e-4)
step(
    "Noisy z constraints applied: "
    f"zl_min={zl_obs_noisy_fp.min():.4f}, zs_min={zs_obs_noisy_fp.min():.4f}, "
    f"min(zs-zl)={(zs_obs_noisy_fp - zl_obs_noisy_fp).min():.4e}"
)

gamma_err_fp = np.full(zl_fp.size, 0.2)
n_gamma_obs = min(10000, zl_fp.size)
gamma_has_obs = np.zeros(zl_fp.size, dtype=bool)
gamma_obs_idx = rng_np.choice(zl_fp.size, size=n_gamma_obs, replace=False)
gamma_has_obs[gamma_obs_idx] = True
gamma_obs_clean_fp = np.zeros(zl_fp.size)
gamma_obs_clean_fp[gamma_has_obs] = gamma_true_fp[gamma_has_obs]
gamma_obs_noisy_fp = np.zeros(zl_fp.size)
gamma_obs_noisy_fp[gamma_has_obs] = gamma_true_fp[gamma_has_obs] + rng_np.normal(0.0, gamma_err_fp[gamma_has_obs])


def build_data(zl_obs, zs_obs, theta_E_obs, vel_obs, gamma_obs):
    return {
        "zl": zl_obs,
        "zs": zs_obs,
        "zl_true": zl_fp,
        "zs_true": zs_fp,
        "zL_err": zL_sigma_fp,
        "zS_err": zS_sigma_fp,
        "theta_E": theta_E_obs,
        "theta_E_true": thetaE_true_fp,
        "theta_E_err": thetaE_err_fp,
        "re": re_fp,
        "vel_obs": vel_obs,
        "vel_err": vel_err_fp,
        "gamma_obs": gamma_obs,
        "gamma_true": gamma_true_fp,
        "gamma_err": gamma_err_fp,
        "gamma_has_obs": gamma_has_obs,
        "beta_true": beta_true_fp,
        "lambda_true": lambda_true_fp,
    }


fp_data_clean = build_data(zl_obs_clean_fp, zs_obs_clean_fp, thetaE_obs_clean_fp, vel_obs_clean_fp, gamma_obs_clean_fp)
fp_data_noisy = build_data(zl_obs_noisy_fp, zs_obs_noisy_fp, thetaE_obs_noisy_fp, vel_obs_noisy_fp, gamma_obs_noisy_fp)


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


def fundamental_plane_model(fp_data):
    cosmo = cosmology_model("waw0cdm", cosmo_prior, sample_h0=True)

    lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.9, 1.1))
    lambda_sigma = numpyro.sample("lambda_sigma", dist.TruncatedNormal(0.05, 0.5, low=0.0, high=0.2))
    gamma_mean = numpyro.sample("gamma_mean", dist.Uniform(1.4, 2.6))
    gamma_sigma = numpyro.sample("gamma_sigma", dist.TruncatedNormal(0.2, 0.2, low=0.0, high=0.4))
    beta_mean = numpyro.sample("beta_mean", dist.Uniform(-0.3, 0.3))
    beta_sigma = numpyro.sample("beta_sigma", dist.TruncatedNormal(0.2, 0.2, low=0.0, high=0.4))

    n_lens = len(fp_data["zl"])
    with numpyro.plate("fp_redshift", n_lens):
        zL_fp = numpyro.sample(
            "zL_fp_latent",
            dist.TruncatedNormal(fp_data["zl"], fp_data["zL_err"], low=1e-4),
        )
        zS_fp = numpyro.sample(
            "zS_fp_latent",
            dist.TruncatedNormal(fp_data["zs"], fp_data["zS_err"], low=zL_fp + Z_SEP_MIN),
        )
    _dl, ds_fp, dls_fp = tool.dldsdls(zL_fp, zS_fp, cosmo, n=20)

    with numpyro.plate("fundamental_plane", n_lens):
        gamma_fp = numpyro.sample("gamma_fp", dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.4, high=2.6))
        beta_fp = numpyro.sample("beta_fp", dist.TruncatedNormal(beta_mean, beta_sigma, low=-0.4, high=0.4))
        lambda_fp = numpyro.sample("lambda_fp", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
        thetaE_fp = numpyro.sample("thetaE_fp", dist.Normal(fp_data["theta_E"], fp_data["theta_E_err"]))

        v_interp_fp = jampy_interp(thetaE_fp, gamma_fp, fp_data["re"], beta_fp)
        vel_pred_fp = v_interp_fp * jnp.sqrt(ds_fp / dls_fp) * jnp.sqrt(lambda_fp)
        gamma_obs_mask = jnp.asarray(fp_data["gamma_has_obs"])
        gamma_obs_used = jnp.where(gamma_obs_mask, fp_data["gamma_obs"], gamma_fp)
        numpyro.sample(
            "gamma_fp_like",
            dist.Normal(gamma_fp, fp_data["gamma_err"]).mask(gamma_obs_mask),
            obs=gamma_obs_used,
        )
        numpyro.sample(
            "vel_fp_like",
            dist.Normal(vel_pred_fp, fp_data["vel_err"]),
            obs=fp_data["vel_obs"],
        )


def build_init_values(fp_data):
    # Keep initial values strictly inside constrained supports to avoid boundary-induced invalid transforms.
    zl_true = np.asarray(fp_data.get("zl_true", fp_data["zl"]), dtype=np.float64)
    zs_true = np.asarray(fp_data.get("zs_true", fp_data["zs"]), dtype=np.float64)
    theta_true = np.asarray(fp_data.get("theta_E_true", fp_data["theta_E"]), dtype=np.float64)
    gamma_true = np.asarray(fp_data.get("gamma_true", fp_data["gamma_obs"]), dtype=np.float64)
    beta_true = np.asarray(fp_data.get("beta_true", np.zeros_like(theta_true)), dtype=np.float64)
    lambda_true = np.asarray(fp_data.get("lambda_true", np.ones_like(theta_true)), dtype=np.float64)

    zl_true = np.maximum(zl_true, 1e-3)
    zs_true = np.maximum(zs_true, zl_true + Z_SEP_MIN + 1e-3)
    theta_true = np.maximum(theta_true, 1e-3)
    gamma_true = np.clip(gamma_true, 1.401, 2.599)
    beta_true = np.clip(beta_true, -0.399, 0.399)
    lambda_true = np.clip(lambda_true, 0.801, 1.199)

    return {
        "Omegam": jnp.asarray(cosmo_true["Omegam"]),
        "w0": jnp.asarray(cosmo_true["w0"]),
        "wa": jnp.asarray(cosmo_true["wa"]),
        "h0": jnp.asarray(cosmo_true["h0"]),
        "lambda_mean": jnp.asarray(1.0),
        "lambda_sigma": jnp.asarray(0.08),
        "gamma_mean": jnp.asarray(2.0),
        "gamma_sigma": jnp.asarray(0.25),
        "beta_mean": jnp.asarray(0.0),
        "beta_sigma": jnp.asarray(0.25),
        "zL_fp_latent": jnp.asarray(zl_true),
        "zS_fp_latent": jnp.asarray(zs_true),
        "gamma_fp": jnp.asarray(gamma_true),
        "beta_fp": jnp.asarray(beta_true),
        "lambda_fp": jnp.asarray(lambda_true),
        "thetaE_fp": jnp.asarray(theta_true),
    }

def run_mcmc(data, key, tag):
    step(f"Run MCMC for lens fundamental-plane ({tag})")
    if TEST_MODE:
        num_warmup, num_samples, num_chains, chain_method = 200, 200, 2, "sequential"
    else:
        num_warmup, num_samples, num_chains, chain_method = 500, 1500, 4, "vectorized"

    nuts = NUTS(
        fundamental_plane_model,
        target_accept_prob=0.95,
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
    mcmc.run(key, fp_data=data)
    print(f"[{tag}] run uses init_to_value")
    extra = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(extra["diverging"]).sum())
    print(f"[{tag}] divergences: {n_div}")

    posterior = mcmc.get_samples(group_by_chain=True)
    inf_data = az.from_dict(posterior=posterior)
    az.to_netcdf(inf_data, RESULT_DIR / f"lens_fp_{tag}.nc")

    trace_vars = ["h0", "Omegam", "w0", "wa", "lambda_mean", "lambda_sigma", "gamma_mean", "gamma_sigma", "beta_mean", "beta_sigma"]
    trace_vars = [v for v in trace_vars if v in inf_data.posterior and inf_data.posterior[v].ndim == 2]
    if trace_vars:
        trace_axes = az.plot_trace(inf_data, var_names=trace_vars, compact=False)
        trace_fig = np.asarray(trace_axes).ravel()[0].figure
        trace_fig.savefig(FIG_DIR / f"lens_fp_trace_{tag}.pdf", dpi=200, bbox_inches="tight")
        plt.close(trace_fig)
    return inf_data


key = random.PRNGKey(42)
key_clean, key_noisy = random.split(key)

step("Execute clean and noisy runs")
idata_clean = run_mcmc(fp_data_clean, key_clean, "clean")
idata_noisy = run_mcmc(fp_data_noisy, key_noisy, "noisy")

step("Create overlay corner plot")
corner_vars = select_corner_vars(
    idata_clean,
    idata_noisy,
    ["h0", "Omegam", "w0", "wa", "lambda_mean", "lambda_sigma", "gamma_mean", "gamma_sigma", "beta_mean", "beta_sigma"],
)
make_overlay_corner(idata_clean, idata_noisy, corner_vars, FIG_DIR / "lens_fp_corner_overlay.pdf")
