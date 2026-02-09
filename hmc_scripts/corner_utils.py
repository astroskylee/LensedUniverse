#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from corner import corner


LABEL_MAP = {
    "h0": r"$H_0$",
    "Omegam": r"$\Omega_m$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
    "lambda_mean": r"$\mu_\lambda$",
    "lambda_sigma": r"$\sigma_\lambda$",
    "gamma_mean": r"$\mu_\gamma$",
    "gamma_sigma": r"$\sigma_\gamma$",
    "beta_mean": r"$\mu_\beta$",
    "beta_sigma": r"$\sigma_\beta$",
}


def _is_scalar_posterior(idata, var_name):
    return idata.posterior[var_name].ndim == 2


def select_corner_vars(clean_idata, noisy_idata, preferred):
    vars_post = [
        name
        for name in preferred
        if (name in clean_idata.posterior) and (name in noisy_idata.posterior)
        and _is_scalar_posterior(clean_idata, name) and _is_scalar_posterior(noisy_idata, name)
    ]
    if vars_post:
        return vars_post
    return [
        name
        for name in clean_idata.posterior.data_vars
        if (name in noisy_idata.posterior)
        and _is_scalar_posterior(clean_idata, name) and _is_scalar_posterior(noisy_idata, name)
    ]


def _combined_ranges(clean_idata, noisy_idata, vars_post):
    ranges = []
    for name in vars_post:
        clean = np.asarray(clean_idata.posterior[name]).reshape(-1)
        noisy = np.asarray(noisy_idata.posterior[name]).reshape(-1)
        merged = np.concatenate([clean, noisy])
        q_lo, q_hi = np.quantile(merged, [0.005, 0.995])
        width = q_hi - q_lo
        pad = 0.1 * width if width > 0 else 1e-6
        ranges.append((q_lo - pad, q_hi + pad))
    return ranges


def make_overlay_corner(clean_idata, noisy_idata, vars_post, outfile):
    if not vars_post:
        return
    labels_post = [LABEL_MAP.get(v, v) for v in vars_post]
    ranges = _combined_ranges(clean_idata, noisy_idata, vars_post)

    fig = corner(
        clean_idata,
        var_names=vars_post,
        labels=labels_post,
        color="#2f8aed",
        show_titles=False,
        title_fmt=".3f",
        levels=[0.68, 0.95],
        fill_contours=True,
        plot_datapoints=False,
        smooth=0.2,
        use_math_text=True,
        label_kwargs=dict(fontsize=15),
        title_kwargs=dict(fontsize=15),
        contour_kwargs={"linewidths": 3},
        hist_kwargs={"density": True, "linewidth": 3},
        range=ranges,
    )
    corner(
        noisy_idata,
        fig=fig,
        var_names=vars_post,
        labels=labels_post,
        color="#f48c06",
        show_titles=False,
        title_fmt=".3f",
        levels=[0.68, 0.95],
        fill_contours=True,
        plot_datapoints=False,
        smooth=0.2,
        use_math_text=True,
        label_kwargs=dict(fontsize=15),
        title_kwargs=dict(fontsize=15),
        contour_kwargs={"linewidths": 3},
        hist_kwargs={"density": True, "linewidth": 3},
        range=ranges,
    )

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
