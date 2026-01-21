# %% imports & setup
import os
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

from SLCOSMO import SLCOSMO, SLmodel, tool

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
zs2_true_cat = data_dspl[:, 2]   # 目录里给的“真值/基准值”，我们当作 true

beta_err_dspl = data_dspl[:, 6]
model_vel_dspl = data_dspl[:, 11]

# --- 强制目录层面 zs2 > zs1（否则直接剔除）
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

zs2_err = np.where(is_photo, 0.1, 1e-4)  # spec-z 给一个很小误差近似固定
zs2_obs = zs2_true_cat + rng_np.normal(0.0, zs2_err)

# --- 强制 zs2_obs > zs1（对 photo-z 那部分做截断/重采样；spec-z 基本不会触发）
eps = 1e-3
bad = zs2_obs <= (zs1_dspl + eps)
# 简单重采样若干次（通常很快收敛）
for _ in range(20):
    if not np.any(bad):
        break
    zs2_obs[bad] = zs2_true_cat[bad] + rng_np.normal(0.0, zs2_err[bad])
    bad = zs2_obs <= (zs1_dspl + eps)
# 最后兜底 clip（极少数还 bad 的）
zs2_obs = np.maximum(zs2_obs, zs1_dspl + eps)

# --- 你原来的距离/β/lambda/vel mock 生成仍然可以基于 true z2（zs2_true_cat）
Dl1, Ds1, Dls1 = tool.compute_distances(zl_dspl, zs1_dspl, cosmo_true)
Dl2, Ds2, Dls2 = tool.compute_distances(zl_dspl, zs2_true_cat, cosmo_true)
beta_geom_dspl = Dls1 * Ds2 / (Ds1 * Dls2)

lambda_true_dspl = tool.truncated_normal(1.0, 0.05, 0.85, 1.15, N_dspl, random_state=rng_np)
lambda_err_dspl = lambda_true_dspl * 0.06
lambda_obs_dspl = lambda_true_dspl + np.random.normal(0.0, lambda_err_dspl)

true_vel_dspl = model_vel_dspl * jnp.sqrt(lambda_true_dspl)
vel_err_dspl = 0.03 * true_vel_dspl
obs_vel_dspl = true_vel_dspl + np.random.normal(0.0, vel_err_dspl)

beta_true_dspl = tool.beta_antimst(beta_geom_dspl, mst=lambda_true_dspl)
beta_obs_dspl = tool.truncated_normal(beta_true_dspl, beta_err_dspl, 0.0, 1.0, random_state=rng_np)

dspl_data = {
    "zl": zl_dspl,
    "zs1": zs1_dspl,
    "zs2_cat": zs2_true_cat,
    "zs2_obs": zs2_obs,
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

dspl_data = {
    "zl": zl_dspl,
    "zs1": zs1_dspl,
    "zs2_cat": zs2_true_cat,
    "zs2_obs": zs2_true_cat,
    "zs2_err": zs2_err,
    "is_photo": is_photo.astype(np.int32),
    "beta_obs": beta_true_dspl,
    "beta_err": beta_err_dspl,
    "v_model": model_vel_dspl,
    "v_obs": obs_vel_dspl,
    "v_err": vel_err_dspl,
    "lambda_err": lambda_err_dspl,
    "lambda_obs": lambda_true_dspl,
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

gamma_obs_lens = gamma_true_lens + tool.truncated_normal(0.0, 0.05, -0.2, 0.2, N_lens, random_state=rng_np)
theta_E_err = 0.01*thetaE_lens
thetaE_lens_obs = thetaE_lens + np.random.normal(0.0, theta_E_err)
#5% on velocity dispersion on gg lens
vel_err_lens = 0.10 * vel_true_lens
vel_obs_lens = np.random.normal(vel_true_lens, vel_err_lens)

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
lens_data = {
    "zl": zl_lens,
    "zs": zs_lens,
    "theta_E": thetaE_lens,
    "theta_E_err": theta_E_err,
    "re": re_lens,
    "gamma_obs": gamma_true_lens,
    "vel_obs": vel_true_lens,
    "vel_err": vel_err_lens,
}
# %% 3) Lensed SNe mock data (Ddt + λ_int constraints)
sn_data = pd.read_csv(os.path.join(DATA_DIR, "Euclid_150SNe.csv"))
sn_data = sn_data[sn_data["tmax"] >= 5][sn_data["tmax"] <= 80]
sn_data = sn_data.nlargest(70, 'tmax')
zl_sne = np.array(sn_data["zl"])
zs_sne = np.array(sn_data["z_host"])
tmax_sne = np.array(sn_data["tmax"])

Dl_sne, Ds_sne, Dls_sne = tool.dldsdls(zl_sne, zs_sne, cosmo_true, n=20)
Ddt_geom_sne = (1.0 + zl_sne) * Dl_sne * Ds_sne / Dls_sne

N_sne = len(zl_sne)
lambda_true_sne = tool.truncated_normal(1.0, 0.05, 0.8, 1.2, N_sne, random_state=rng_np)
Ddt_true_sne = Ddt_geom_sne * lambda_true_sne

frac_err_Ddt = np.sqrt((1.0 / tmax_sne) ** 2 + 0.05 ** 2)
Ddt_obs_sne = Ddt_true_sne * np.random.normal(1.0, frac_err_Ddt)
Ddt_err_sne = Ddt_true_sne * frac_err_Ddt

lambda_obs_sne = lambda_true_sne * np.random.normal(1.0, 0.08, N_sne)
lambda_err_sne = 0.08 * lambda_true_sne

sne_data = {
    "zl": zl_sne,
    "zs": zs_sne,
    "Ddt_obs": Ddt_obs_sne,
    "Ddt_err": Ddt_err_sne,
    "lambda_obs": lambda_obs_sne,
    "lambda_err": lambda_err_sne,
}

sne_data = {
    "zl": zl_sne,
    "zs": zs_sne,
    "Ddt_obs": Ddt_true_sne,
    "Ddt_err": Ddt_err_sne,
    "lambda_obs": lambda_true_sne,
    "lambda_err": lambda_err_sne,
}

# %% joint hierarchical model
def joint_model(dspl_data = None, lens_data = None, sne_data = None):
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
        # ----- SNe block -----
        N_sne = len(sne_data["zl"])
        with numpyro.plate("sne", N_sne):
            lambda_sne = numpyro.sample("lambda_sne", dist.Normal(lambda_mean, lambda_sigma))
            numpyro.sample("lambda_sne_like",
                           dist.Normal(lambda_sne, sne_data["lambda_err"]),
                           obs=sne_data["lambda_obs"])
            Ddt_true = Ddt_geom * lambda_sne
            numpyro.sample("Ddt_sne_like",
                           dist.Normal(Ddt_true, sne_data["Ddt_err"]),
                           obs=sne_data["Ddt_obs"])



def head_dict(data_dict, N_use=None):
    # cut dictionary
    return {k: np.asarray(v)[:N_use] for k, v in data_dict.items()}

if TEST_MODE:
    N_DSPL_USE = 50
    N_LENS_USE = 200
    N_SNE_USE = 10
    num_warmup = 200
    num_samples = 200
    num_chains = 1
    chain_method = "sequential"
else:
    N_DSPL_USE = 1200
    N_LENS_USE = 5000
    N_SNE_USE = 50
    num_warmup = 2000
    num_samples = 5000
    num_chains = 8
    chain_method = "vectorized"

dspl_data = head_dict(dspl_data, N_DSPL_USE)
lens_data = head_dict(lens_data, N_LENS_USE)
sne_data  = head_dict(sne_data,  N_SNE_USE)

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

if TEST_MODE:
    nc_filename = "combine_forecast_test_output"
else:
    nc_filename = "/mnt/lustre/tianli/slcosmo_result/Lens_revolution"
print(az.summary(inf_data, var_names=corner_vars)) 
summary_df = az.summary(inf_data, var_names=corner_vars)
csv_path = nc_filename+ "_summary.csv"
summary_df.to_csv(csv_path)
print("Saved:", csv_path)

az.to_netcdf(inf_data, nc_filename+".nc") 
print(f"Saved InferenceData to {nc_filename}") 

if not TEST_MODE:
    fig = corner.corner(inf_data, truths=truths, var_names=corner_vars, show_title=True)
    fig.savefig(nc_filename + ".pdf")
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

    g.export(nc_filename + "compare.pdf")  # 等价于保存当前 g.fig

    # 关键：显式释放/关闭，避免 __del__ 在解释器退出时炸
    plt.close(g.fig)
    del g
    import gc

    gc.collect()
