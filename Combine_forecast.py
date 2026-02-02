# %% imports & setup
import os
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import arviz as az

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from jax import random

from slcosmo import SLCOSMO, SLmodel, tool

TEST_MODE = os.environ.get("COMBINE_FORECAST_TEST") == "1"
DATA_DIR = os.environ.get("SLCOSMO_DATA_DIR", os.path.join("..", "slcosmo"))
OTHER_FORECAST_DIR = os.environ.get("OTHER_FORECAST_DIR", os.path.join("..", "SLCOSMO", "other_forecast"))

jax.config.update("jax_enable_x64", True)
if TEST_MODE:
    print("Test mode: using GPU")
    numpyro.set_platform("gpu")
else:
    print("GPU connected")
    numpyro.set_platform("gpu")
numpyro.enable_x64()

slcosmo = SLCOSMO()
model_instance = SLmodel(slcosmo)

SEED = 42
rng_np = np.random.default_rng(SEED)   # 统一 mock 随机源
np.random.seed(SEED)                  # 可选兜底：如果你漏改了某个 np.random 调用

# ---------------------------
# Noisy vs noise-free options (per subprobe)
def _env_flag(key, default="0"):
    return os.environ.get(key, default) == "1"

USE_NOISY_DSPL = _env_flag("COMBINE_NOISY_DSPL")
USE_NOISY_LENS = _env_flag("COMBINE_NOISY_LENS")
USE_NOISY_SNE = _env_flag("COMBINE_NOISY_SNE")
USE_NOISY_QUASAR = _env_flag("COMBINE_NOISY_QUASAR")


# %% cosmology model & priors
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

cosmo_prior = {
    "w0_up": 0.0,   "w0_low": -2.0,
    "wa_up": 2.0,   "wa_low": -2.0,
    "omegak_up": 1.0, "omegak_low": -1.0,
    "h0_up": 80.0,  "h0_low": 60.0,
    "omegam_up": 0.5, "omegam_low": 0.1,
}

# mock “true” cosmology for data generation
cosmo_true = {"Omegam": 0.32, "Omegak": 0.0, "w0": -1.0, "wa": 0.0, "h0": 70.0}

# %% 1) DSPL mock data (β + σ_v)  + 60% photo-z on zs2
data_dspl = np.loadtxt(os.path.join(DATA_DIR, "EuclidDSPLs_1.txt"))
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
print("N_dspl after zs2>zs1 cut =", N_dspl)

# --- 60% systems use photo-z on zs2 with sigma=0.1
is_photo = (rng_np.random(N_dspl) < 0.60)
zs2_err = np.where(is_photo, 0.1, 1e-4)
zs2_obs = zs2_true_cat + rng_np.normal(0.0, zs2_err)

# --- enforce zs2_obs > zs1
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

lambda_true_dspl = tool.truncated_normal(1.0, 0.05, 0.85, 1.15, N_dspl, random_state=rng_np)
lambda_err_dspl = lambda_true_dspl * 0.06

true_vel_dspl = model_vel_dspl * jnp.sqrt(lambda_true_dspl)
vel_err_dspl = 0.03 * true_vel_dspl

beta_true_dspl = tool.beta_antimst(beta_geom_dspl, mst=lambda_true_dspl)

if USE_NOISY_DSPL:
    lambda_obs_dspl = lambda_true_dspl + np.random.normal(0.0, lambda_err_dspl)
    obs_vel_dspl = true_vel_dspl + np.random.normal(0.0, vel_err_dspl)
    beta_obs_dspl = tool.truncated_normal(beta_true_dspl, beta_err_dspl, 0.0, 1.0, random_state=rng_np)
    zs2_use = zs2_obs
else:
    lambda_obs_dspl = lambda_true_dspl
    obs_vel_dspl = true_vel_dspl
    beta_obs_dspl = beta_true_dspl
    zs2_use = zs2_true_cat

dspl_data = {
    "zl": zl_dspl,
    "zs1": zs1_dspl,
    "zs2_cat": zs2_true_cat,
    "zs2_obs": zs2_use,
    "zs2_err": zs2_err,
    "is_photo": is_photo.astype(np.int32),
    "beta_obs": beta_obs_dspl,
    "beta_err": beta_err_dspl,
    "v_model": model_vel_dspl,
    "v_obs": obs_vel_dspl,
    "v_err": vel_err_dspl,
    "lambda_err": lambda_err_dspl,
    "lambda_obs": lambda_obs_dspl,
}

photo_z = True
# %% 2) Lens+kin mock data (E, γ, β_aniso, σ_v)
LUT = np.load(os.path.join(DATA_DIR, "velocity_disp_table.npy"))
N1, N2, N3, N4 = LUT.shape
thetaE_grid = np.linspace(0.5, 3.0, N1)
gamma_grid  = np.linspace(1.2, 2.8, N2)
Re_grid     = np.linspace(0.15, 3.0, N3)
beta_grid   = np.linspace(-0.5, 0.8, N4)
jampy_interp = tool.make_4d_interpolant(thetaE_grid, gamma_grid, Re_grid, beta_grid, LUT)

Euclid_GG_data = np.loadtxt(os.path.join(DATA_DIR, "Euclid_len.txt"))
zl_lens = Euclid_GG_data[:, 0]
zs_lens = Euclid_GG_data[:, 1]
Ein_lens = Euclid_GG_data[:, 2]
re_lens = Euclid_GG_data[:, 5]

mask_lens = (Ein_lens >= 0.6) & (re_lens >= 0.25) & (re_lens <= 2.8)
zl_lens = zl_lens[mask_lens]
zs_lens = zs_lens[mask_lens]
thetaE_lens = Ein_lens[mask_lens]
re_lens = re_lens[mask_lens]

dl_lens, ds_lens, dls_lens = tool.dldsdls(zl_lens, zs_lens, cosmo_true, n=20)
N_lens = len(zl_lens)

gamma_true_lens = tool.truncated_normal(2.0, 0.2, 1.5, 2.5, N_lens, random_state=rng_np)
beta_true_lens  = tool.truncated_normal(0.0, 0.2, -0.4, 0.4, N_lens, random_state=rng_np)
vel_model_lens = jampy_interp(thetaE_lens, gamma_true_lens, re_lens, beta_true_lens) * jnp.sqrt(ds_lens / dls_lens)
lambda_true_lens = tool.truncated_normal(1.0, 0.05, 0.8, 1.2, N_lens, random_state=rng_np)
vel_true_lens = vel_model_lens * jnp.sqrt(lambda_true_lens)

theta_E_err = 0.01 * thetaE_lens
vel_err_lens = 0.10 * vel_true_lens

if USE_NOISY_LENS:
    gamma_obs_lens = gamma_true_lens + tool.truncated_normal(0.0, 0.05, -0.2, 0.2, N_lens, random_state=rng_np)
    thetaE_lens_obs = thetaE_lens + np.random.normal(0.0, theta_E_err)
    vel_obs_lens = np.random.normal(vel_true_lens, vel_err_lens)
else:
    gamma_obs_lens = gamma_true_lens
    thetaE_lens_obs = thetaE_lens
    vel_obs_lens = vel_true_lens

lens_data = {
    "zl": zl_lens,
    "zs": zs_lens,
    "theta_E": thetaE_lens_obs,
    "theta_E_err": theta_E_err,
    "re": re_lens,
    "gamma_obs": gamma_obs_lens,
    "vel_obs": vel_obs_lens,
    "vel_err": vel_err_lens,
}
# %% 3) Lensed SNe mock data (time-delay likelihood)
sn_data = pd.read_csv(os.path.join(DATA_DIR, "Euclid_150SNe.csv"))
sn_data = sn_data[(sn_data["tmax"] >= 5) & (sn_data["tmax"] <= 80)]
sn_data = sn_data.nlargest(70, 'tmax')
zl_sne = np.array(sn_data["zl"])
zs_sne = np.array(sn_data["z_host"])
t_delay_true_days = np.array(sn_data["tmax"])

Dl_sne, Ds_sne, Dls_sne = tool.dldsdls(zl_sne, zs_sne, cosmo_true, n=20)
Ddt_geom_sne = (1.0 + zl_sne) * Dl_sne * Ds_sne / Dls_sne

N_sne = len(zl_sne)
# Parent population for MST (lambda)
lambda_pop_mean = 1.0
lambda_pop_sigma = 0.05
lambda_low, lambda_high = 0.8, 1.2
lambda_true_sne = tool.truncated_normal(lambda_pop_mean, lambda_pop_sigma, lambda_low, lambda_high, N_sne, random_state=rng_np)

seconds_per_day = 86400.0
Mpc_km = tool.Mpc / 1000.0

# Fermat potential difference from baseline time delay (no MST in phi)
fermat_phi_true = (tool.c_km_s * t_delay_true_days * seconds_per_day) / (Ddt_geom_sne * Mpc_km)

# Time-delay true includes MST transform
t_delay_true_mst = t_delay_true_days * lambda_true_sne

# Measurement errors (assumed in likelihood)
sigma_t_days = 1.0
sigma_phi_frac = 0.04
sigma_lambda_frac = 0.08

# Observed quantities (noisy vs noise-free)
if USE_NOISY_SNE:
    t_delay_obs = t_delay_true_mst + np.random.normal(0.0, sigma_t_days, size=N_sne)
    fermat_phi_obs = fermat_phi_true + np.random.normal(0.0, sigma_phi_frac * np.abs(fermat_phi_true))
    lambda_obs_sne = lambda_true_sne + np.random.normal(0.0, sigma_lambda_frac * np.abs(lambda_true_sne))
else:
    t_delay_obs = t_delay_true_mst.copy()
    fermat_phi_obs = fermat_phi_true.copy()
    lambda_obs_sne = lambda_true_sne.copy()

lambda_err_sne = sigma_lambda_frac * np.abs(lambda_obs_sne)

# Scale phi
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

fermat_phi_obs_scaled, phi_scale_sne = scale_phi(fermat_phi_obs)

sne_data = {
    "zl": zl_sne,
    "zs": zs_sne,
    "t_obs": t_delay_obs,
    "phi_obs": fermat_phi_obs_scaled,
    "phi_scale": phi_scale_sne,
    "lambda_obs": lambda_obs_sne,
    "lambda_err": lambda_err_sne,
}
# %% 4) Lensed Quasar mock data (time-delay likelihood)
DATA_JSON = Path("../Temp_data/static_datavectors_seed6.json")
with DATA_JSON.open("r") as f:
    quasar_blocks = json.load(f)

z_lens_list = []
z_src_list = []
t_base_list = []
t_err_list = []
block_id_list = []

for b, block in enumerate(quasar_blocks):
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

z_lens_q = np.concatenate(z_lens_list)
z_src_q = np.concatenate(z_src_list)
t_base_q = np.concatenate(t_base_list)
t_err_q = np.concatenate(t_err_list)
block_id_q = np.concatenate(block_id_list)

zl_j = jnp.asarray(z_lens_q)
zs_j = jnp.asarray(z_src_q)
Dl_q, Ds_q, Dls_q = tool.dldsdls(zl_j, zs_j, cosmo_true, n=20)
Ddt_geom_q = (1.0 + zl_j) * Dl_q * Ds_q / Dls_q
Ddt_geom_q = np.asarray(Ddt_geom_q)

c_km_day = tool.c_km_s * 86400.0
Mpc_km = tool.Mpc / 1000.0

phi_true_q = (c_km_day * t_base_q) / (Ddt_geom_q * Mpc_km)

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
    3: 0.05,
    4: 0.05,
    5: 0.05,
    6: 0.05,
    7: np.nan,
    8: np.nan,
}

phi_err_frac_q = np.asarray([phi_err_frac_by_block[b] for b in block_id_q])
phi_err_q = phi_err_frac_q * np.abs(phi_true_q)

lambda_true_q = tool.truncated_normal(1.0, 0.05, 0.8, 1.2, z_lens_q.size, random_state=rng_np)

sigma_v_frac_q = np.asarray([sigma_v_frac_by_block[b] for b in block_id_q])
mst_mask_q = np.isfinite(sigma_v_frac_q)
mst_err_frac_q = 2.0 * sigma_v_frac_q
lambda_err_q = np.where(
    np.isfinite(mst_err_frac_q),
    mst_err_frac_q * np.abs(lambda_true_q),
    0.05,
)

t_true_q = t_base_q * lambda_true_q

if USE_NOISY_QUASAR:
    t_obs_q = t_true_q + rng_np.normal(0.0, t_err_q)
    phi_obs_q = phi_true_q + rng_np.normal(0.0, phi_err_q)
    lambda_obs_q = lambda_true_q + rng_np.normal(0.0, lambda_err_q)
else:
    t_obs_q = t_true_q.copy()
    phi_obs_q = phi_true_q.copy()
    lambda_obs_q = lambda_true_q.copy()

phi_obs_q_scaled, phi_scale_q = scale_phi(phi_obs_q)
phi_err_q_scaled = phi_err_frac_q * np.abs(phi_obs_q_scaled)

quasar_data = {
    "zl": z_lens_q,
    "zs": z_src_q,
    "t_obs": t_obs_q,
    "t_err": t_err_q,
    "phi_obs": phi_obs_q_scaled,
    "phi_err": phi_err_q_scaled,
    "phi_scale": phi_scale_q,
    "lambda_obs": lambda_obs_q,
    "lambda_err": lambda_err_q,
    "mst_mask": mst_mask_q,
}
# %% joint hierarchical model
def joint_model(dspl_data = None, lens_data = None, sne_data = None, quasar_data = None):
    # shared cosmology
    cosmo = cosmology_model("waw0cdm", cosmo_prior, sample_h0=True)
    # shared MST population
    lambda_mean = numpyro.sample("lambda_mean", dist.Uniform(0.9, 1.1))
    lambda_sigma = numpyro.sample("lambda_sig", dist.TruncatedNormal(0.05, 0.5, low=0.0, high=0.2))

    # lens slope & anisotropy population (only used in lens+kin block)
    gamma_mean = numpyro.sample("gamma_mean", dist.Uniform(1.8, 2.2))
    gamma_sigma = numpyro.sample("gamma_sigma", dist.TruncatedNormal(0.2, 0.5, low=0.0, high=0.4))
    beta_mean  = numpyro.sample("beta_mean", dist.Uniform(-0.1, 0.1))
    beta_sigma = numpyro.sample("beta_sigma", dist.TruncatedNormal(0.2, 0.5, low=0.0, high=0.4))

    # ----- DSPL block -----
    if dspl_data is not None:
        N_dspl = len(dspl_data["zl"])
    
        zl  = jnp.asarray(dspl_data["zl"])
        zs1 = jnp.asarray(dspl_data["zs1"])
        zs2_obs = jnp.asarray(dspl_data["zs2_obs"])
        zs2_err = jnp.asarray(dspl_data["zs2_err"])
    
        # zs1 distances（确定量）
        Dl1, Ds1, Dls1 = tool.compute_distances(zl, zs1, cosmo)
    
        # 关键：采样每个系统的 zs2_true（向量化，一次 sample）
        if photo_z:
            eps = 1e-3
            zs2_true = numpyro.sample(
                "zs2_true",
                dist.TruncatedNormal(zs2_obs, zs2_err, low=zs1 + eps, high=10.0).to_event(1)
            )
        
            # 用采样出来的 zs2_true 计算距离与 beta
        else:
            zs2_true = dspl_data['zs2_cat']
        Dl2, Ds2, Dls2 = tool.compute_distances(zl, zs2_true, cosmo)
        beta_geom = Dls1 * Ds2 / (Ds1 * Dls2)
    
        with numpyro.plate("dspl", N_dspl):
            lambda_dspl = numpyro.sample(
                "lambda_dspl",
                dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2),
            )
    
            numpyro.sample("lambda_dspl_like",
                           dist.Normal(lambda_dspl, jnp.asarray(dspl_data["lambda_err"])),
                           obs=jnp.asarray(dspl_data["lambda_obs"]))
    
            beta_mst = tool.beta_antimst(beta_geom, lambda_dspl)
            numpyro.sample("beta_dspl_like",
                           dist.TruncatedNormal(beta_mst, jnp.asarray(dspl_data["beta_err"]), low=0.0, high=1.0),
                           obs=jnp.asarray(dspl_data["beta_obs"]))


    # ----- lens+kin block -----
    if lens_data is not None:
        # lens+kin distances
        dl_lens, ds_lens, dls_lens = tool.dldsdls(lens_data["zl"], lens_data["zs"], cosmo, n=20)
        N_lens = len(lens_data["zl"])
        with numpyro.plate("lens", N_lens):
            gamma_i = numpyro.sample(
                "gamma_i",
                dist.TruncatedNormal(gamma_mean, gamma_sigma, low=1.6, high=2.4),
            )
            beta_i = numpyro.sample(
                "beta_i",
                dist.TruncatedNormal(beta_mean, beta_sigma, low=-0.4, high=0.4),
            )
            lambda_lens = numpyro.sample(
                "lambda_lens",
                dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2),
            )
            theta_E_i = numpyro.sample(
                "theta_E_i",
                dist.Normal(lens_data["theta_E"], lens_data["theta_E_err"]),
            )
            v_interp = jampy_interp(theta_E_i, gamma_i, lens_data["re"], beta_i)
            vel_pred = v_interp * jnp.sqrt(ds_lens / dls_lens) * jnp.sqrt(lambda_lens)
    
            numpyro.sample("gamma_obs_lens",
                           dist.Normal(gamma_i, 0.05),
                           obs=lens_data["gamma_obs"])
            numpyro.sample("vel_lens_like",
                           dist.Normal(vel_pred, lens_data["vel_err"]),
                           obs=lens_data["vel_obs"])

    if sne_data is not None:
        # SNe distances
        Dl_sne, Ds_sne, Dls_sne = tool.dldsdls(sne_data["zl"], sne_data["zs"], cosmo, n=20)
        Ddt_geom = (1.0 + sne_data["zl"]) * Dl_sne * Ds_sne / Dls_sne
        N_sne = len(sne_data["zl"])

        t_obs = sne_data["t_obs"]
        phi_obs = sne_data["phi_obs"]
        phi_scale = sne_data["phi_scale"]
        lambda_obs = sne_data["lambda_obs"]
        lambda_err = sne_data["lambda_err"]
        sigma_phi = sigma_phi_frac * phi_obs

        with numpyro.plate("sne", N_sne):
            phi_true_scaled = numpyro.sample("phi_true_scaled_sne", dist.TruncatedNormal(phi_obs, sigma_phi, low=0.0, high=10.0))
            lambda_sne = numpyro.sample("lambda_sne", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
            numpyro.sample(
                "lambda_sne_like",
                dist.Normal(lambda_sne, lambda_err),
                obs=lambda_obs,
            )

            phi_true = phi_true_scaled / phi_scale
            Ddt_true = Ddt_geom * lambda_sne
            t_model_days = (Ddt_true * Mpc_km / tool.c_km_s) * phi_true / seconds_per_day
            numpyro.sample(
                "t_delay_sne_like",
                dist.Normal(t_model_days, sigma_t_days),
                obs=t_obs,
            )


    if quasar_data is not None:
        Dl_q, Ds_q, Dls_q = tool.dldsdls(quasar_data["zl"], quasar_data["zs"], cosmo, n=20)
        Ddt_geom_q = (1.0 + quasar_data["zl"]) * Dl_q * Ds_q / Dls_q
        N_q = len(quasar_data["zl"])

        t_obs = quasar_data["t_obs"]
        t_err = quasar_data["t_err"]
        phi_obs = quasar_data["phi_obs"]
        phi_err = quasar_data["phi_err"]
        phi_scale = quasar_data["phi_scale"]
        lambda_obs = quasar_data["lambda_obs"]
        lambda_err = quasar_data["lambda_err"]
        mst_mask = jnp.asarray(quasar_data["mst_mask"])

        with numpyro.plate("quasar", N_q):
            phi_true_scaled = numpyro.sample("phi_true_scaled_q", dist.Normal(phi_obs, phi_err))
            lambda_q = numpyro.sample("lambda_q", dist.TruncatedNormal(lambda_mean, lambda_sigma, low=0.8, high=1.2))
            numpyro.sample("lambda_q_like", dist.Normal(lambda_q, lambda_err).mask(mst_mask), obs=lambda_obs)

            phi_true = phi_true_scaled / phi_scale
            Ddt_true = Ddt_geom_q * lambda_q
            t_model_days = (Ddt_true * Mpc_km / tool.c_km_s) * phi_true / seconds_per_day
            numpyro.sample("t_delay_q_like", dist.Normal(t_model_days, t_err), obs=t_obs)



def head_dict(data_dict, N_use=None):
    out = {}
    for k, v in data_dict.items():
        arr = np.asarray(v)
        if arr.shape == ():
            out[k] = arr
        else:
            out[k] = arr[:N_use] if N_use is not None else arr
    return out

if TEST_MODE:
    N_DSPL_USE = 50
    N_LENS_USE = 200
    N_SNE_USE = 10
    N_QUASAR_USE = 30
    num_warmup = 200
    num_samples = 200
    num_chains = 2
    chain_method = "sequential"
else:
    N_DSPL_USE = 1200
    N_LENS_USE = 5000
    N_SNE_USE = 50
    N_QUASAR_USE = 500
    num_warmup = 2000
    num_samples = 5000
    num_chains = 8
    chain_method = "vectorized"

dspl_data = head_dict(dspl_data, N_DSPL_USE)
lens_data = head_dict(lens_data, N_LENS_USE)
sne_data  = head_dict(sne_data,  N_SNE_USE)
quasar_data = head_dict(quasar_data, N_QUASAR_USE)

init_values = {
    "h0": 70.0,
    "Omegam": 0.32,
    "w0": -1.0,
    "wa": 0.0,
    "lambda_mean": 1.0,
    "lambda_sig": 0.05,
    # 也可以给 hyper-sigma、gamma_mean 等加上
    "gamma_mean": 2.0,
    "gamma_sigma": 0.2,
    "beta_mean": 0.0,
    "beta_sigma": 0.2,
}
from numpyro.infer import init_to_value, init_to_median
init_strategy = init_to_value(values=init_values)

nuts_kernel = NUTS(joint_model, target_accept_prob=0.8, dense_mass=[('wa', 'w0', 'h0', 'Omegam', 'lambda_mean')], init_strategy=init_strategy)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method=chain_method,
    progress_bar=True,
)

rng_key = random.PRNGKey(0)
corner_vars = [
    "h0", "Omegam", "w0", "wa",
    "gamma_mean", "gamma_sigma",
    "beta_mean", "beta_sigma",
    "lambda_mean", "lambda_sig",
]
import corner
truths = [70, 0.32, -1.0, 0.0, 2.0, 0.2, 0.0, 0.2, 1.0, 0.05]

mcmc.run(rng_key,dspl_data=dspl_data, sne_data = sne_data, lens_data = lens_data) 

posterior = jax.device_get(mcmc.get_samples(group_by_chain=True))
sample_stats = jax.device_get(mcmc.get_extra_fields(group_by_chain=True))
inf_data = az.from_dict(
    posterior=posterior,
    sample_stats=sample_stats,   # 可按需筛选字段
)

result_dir = "/mnt/lustre/tianli/LensedUniverse_result"
os.makedirs(result_dir, exist_ok=True)

if TEST_MODE:
    nc_filename = os.path.join(result_dir, "combine_forecast_test_output")
else:
    nc_filename = os.path.join(result_dir, "Lens_revolution")
print(az.summary(inf_data, var_names=corner_vars)) 
summary_df = az.summary(inf_data, var_names=corner_vars)
csv_path = nc_filename+ "_summary.csv"
summary_df.to_csv(csv_path)
print("Saved:", csv_path)

az.to_netcdf(inf_data, nc_filename+".nc") 
print(f"Saved InferenceData to {nc_filename}") 

fig_dir = Path("result")
fig_dir.mkdir(parents=True, exist_ok=True)
base_name = os.path.basename(nc_filename)

if not TEST_MODE:
    fig = corner.corner(inf_data, truths=truths, var_names=corner_vars, show_title=True)
    fig.savefig(fig_dir / f"{base_name}.pdf")
    plt.close(fig)          # 关键：立刻关闭


if not TEST_MODE:
    import numpy as np
    import pandas as pd
    import arviz as az
    import matplotlib.pyplot as plt

    from getdist import plots, MCSamples

    # ============================
    # 1. 读入三套 w0, wa 样本
    # ============================

    # (1) LSST SNe：Y10_SN.txt
    posterior_supernovae = pd.DataFrame(
        np.genfromtxt(os.path.join(OTHER_FORECAST_DIR, "Y10_SN.txt")),
        columns=["Omega_m", "sigma_8", "n_s", "w_0", "w_a", "Omega_b", "H_0"],
    )
    # 只要 w0, wa
    samples_supernovae = posterior_supernovae[["w_0", "w_a"]].to_numpy()

    # (2) Euclid 弱透镜：Fisher -> 高斯样本
    param_names_all = [
        "Omegam",
        "Omegab",
        "w0",
        "wa",
        "h",
        "ns",
        "sigma8",
        "aIA",
        "etaIA",
        "betaIA",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "b8",
        "b9",
        "b10",
    ]

    fisher_matrix = np.genfromtxt(
        os.path.join(OTHER_FORECAST_DIR, "EuclidISTF_WL_w0wa_flat_pessimistic.txt")
    )
    covariance_matrix = np.linalg.inv(fisher_matrix)

    idx_w0 = param_names_all.index("w0")
    idx_wa = param_names_all.index("wa")
    cov_sub = covariance_matrix[np.ix_([idx_w0, idx_wa], [idx_w0, idx_wa])]

    mean_w0 = -1.0
    mean_wa = 0.0
    mean = [mean_w0, mean_wa]

    num_samples = 10000
    samples_Euclid = np.random.multivariate_normal(mean, cov_sub, size=num_samples)

    # (3) 你的 joint MST + cosmology 后验：joint_mst_cosmology.nc
    # idata = az.from_netcdf("joint_mst_cosmology.nc")
    idata = az.from_netcdf(nc_filename + ".nc")
    w0_joint = idata.posterior["w0"].values.reshape(-1)
    wa_joint = idata.posterior["wa"].values.reshape(-1)
    w0_joint = w0_joint - np.mean(w0_joint) - 1
    wa_joint = wa_joint - np.mean(wa_joint)
    samples_joint = np.vstack([w0_joint, wa_joint]).T

    # ============================
    # 2. 转成 GetDist 的 MCSamples
    # ============================

    names = ["w0", "wa"]
    labels = [r"w_0", r"w_a"]

    mc_sne = MCSamples(
        samples=samples_supernovae,
        names=names,
        labels=labels,
        settings={"smooth_scale_2D": 0.5},
    )

    mc_euclid = MCSamples(
        samples=samples_Euclid,
        names=names,
        labels=labels,
    )

    mc_joint = MCSamples(
        samples=samples_joint,
        names=names,
        labels=labels,
        settings={"smooth_scale_2D": 0.5},
    )

    # （可选）给每个样本设置相同的范围，避免尾巴太远
    # for mc in [mc_sne, mc_euclid, mc_joint]:
    #     mc.setRanges({'w0': [-1.3, -0.7], 'wa': [-1.0, 1.0]})

    # ============================
    # 3. GetDist 2D 图
    # ============================

    g = plots.get_subplot_plotter(subplot_size=5)

    g.plot_2d(
        [mc_sne, mc_euclid, mc_joint],
        param_pair=("w0", "wa"),
        filled=[True, True, True],
        line_args=[
            {"ls": "--", "alpha": 0.7, "lw": 2},
            {"ls": "--", "alpha": 0.7, "lw": 2},
            {"ls": "-", "lw": 2},
        ],
        colors=["orange", "green", "black"],
    )

    g.add_legend(
        ["LSST SNe", "Euclid Weak Lensing", "Strong Lensing Y10"],
        frameon=True,
        framealpha=0.8,
        edgecolor="black",
        facecolor="white",
        fontsize=10,
        legend_loc="upper left",
    )

    g.export(str(fig_dir / f"{base_name}compare.pdf"))  # 等价于保存当前 g.fig

    # 关键：显式释放/关闭，避免 __del__ 在解释器退出时炸
    plt.close(g.fig)
    del g
    import gc

    gc.collect()
