#!/usr/bin/env python3
"""
Inspect sigma_v_measured to diagnose NaNs in mst_err.

Logic:
  - For each block, sigma_v_measured is averaged over the image-pair axis.
  - mst_err per lens is defined as std/mean over the chain axis.
  - NaNs arise when sigma_v_measured is non-finite or mean is ~0.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MST error sources from sigma_v_measured.")
    parser.add_argument("--input", default="../Temp_data/static_datavectors_seed6.json", help="Input JSON file")
    parser.add_argument("--eps", type=float, default=1e-12, help="Mean threshold for division safety")
    parser.add_argument("--max-show", type=int, default=10, help="Max bad lens indices to print per block")
    args = parser.parse_args()

    with Path(args.input).open("r") as f:
        data = json.load(f)

    has_mst = np.array([("sigma_v_measured" in d) for d in data])
    missing_blocks = np.where(~has_mst)[0]
    print("Blocks:", len(data))
    print("Blocks without sigma_v_measured:", missing_blocks.tolist())

    print("\nblock  n_lens  n_td  n_obs  has_mst")
    for b, d in enumerate(data):
        z_lens = np.asarray(d["z_lens"])
        td = np.asarray(d["td_measured"])
        n_lens = z_lens.shape[0]
        n_td = td.shape[2]
        n_obs = n_lens * n_td
        print(f"{b:5d} {n_lens:7d} {n_td:4d} {n_obs:6d} {str(bool(has_mst[b])):>7}")

    print("\n--- MST diagnostics ---")
    total_nonfinite = 0
    total_mean_bad = 0
    total_std_bad = 0
    total_mst_bad = 0

    for b in np.where(has_mst)[0]:
        d = data[int(b)]
        sig = np.asarray(d["sigma_v_measured"], dtype=float)
        sig = sig.mean(axis=2)

        sig_finite = np.isfinite(sig)
        n_nonfinite = sig_finite.size - sig_finite.sum()

        sig_mean = np.mean(sig, axis=1)
        sig_std = np.std(sig, axis=1, ddof=1)

        mean_bad = (~np.isfinite(sig_mean)) | (np.abs(sig_mean) < args.eps)
        std_bad = ~np.isfinite(sig_std)

        mst_err = sig_std / sig_mean
        mst_bad = ~np.isfinite(mst_err)

        total_nonfinite += int(n_nonfinite)
        total_mean_bad += int(mean_bad.sum())
        total_std_bad += int(std_bad.sum())
        total_mst_bad += int(mst_bad.sum())

        bad_idx = np.where(mst_bad)[0]
        print(
            f"block {int(b):02d}: nonfinite_sigma_v={int(n_nonfinite)}, "
            f"mean_bad={int(mean_bad.sum())}, std_bad={int(std_bad.sum())}, mst_bad={int(mst_bad.sum())}"
        )
        print("  bad lens idx:", bad_idx[: args.max_show].tolist())

        print("  per-lens stats (mean, std, std/mean):")
        for i in range(sig_mean.shape[0]):
            print(f"    lens {i:03d}: mean={sig_mean[i]:.6e}, std={sig_std[i]:.6e}, ratio={mst_err[i]:.6e}")

    print("\nSummary:")
    print("  total nonfinite sigma_v:", total_nonfinite)
    print("  total mean_bad lenses:", total_mean_bad)
    print("  total std_bad lenses:", total_std_bad)
    print("  total mst_bad lenses:", total_mst_bad)


if __name__ == "__main__":
    main()
