#!/usr/bin/env python3
"""
Prepare quasar time-delay data vectors for cosmology inference.

Inputs:
  - static_datavectors_seed6.json (list of blocks)

Outputs (flat arrays, one row per time-delay measurement):
  - z_lens, z_src
  - fpd_err
  - td_true, td_err
  - mst_true, mst_err
  - block_id, lens_id, pair_id

Notes:
  - fpd/td stats are computed as mean/std across chain axis=1.
  - fpd_true is computed in the notebook from z_lens, z_src, and td_true.
  - MST samples are computed as sigma_v_measured / kin_pred_samples;
    if last axis exists, it is averaged (compressed) to one MST per lens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _mean_std(samples: np.ndarray, axis: int = 1, min_err: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=axis)
    std = np.std(samples, axis=axis, ddof=1)
    if min_err is not None:
        std = np.where(std > 0, std, min_err)
    return mean, std


def _compute_mst_stats(block: Dict, min_err: float) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if ("sigma_v_measured" not in block) or ("kin_pred_samples" not in block):
        return None, None

    sig = np.asarray(block["sigma_v_measured"], dtype=float)
    kin = np.asarray(block["kin_pred_samples"], dtype=float)

    mst_samples = sig / kin

    # Compress any extra dimension (e.g., 3 delays for quad) to one MST per lens.
    if mst_samples.ndim == 3:
        mst_samples = mst_samples.mean(axis=2)

    mst_mean, mst_std = _mean_std(mst_samples, axis=1, min_err=min_err)
    return mst_mean, mst_std


def _flatten_block(
    z_lens: np.ndarray,
    z_src: np.ndarray,
    fpd_std: np.ndarray,
    td_mean: np.ndarray,
    td_std: np.ndarray,
    mst_mean: np.ndarray,
    mst_std: np.ndarray,
    block_id: int,
) -> Dict[str, np.ndarray]:
    n_lens, n_td = td_mean.shape

    z_lens_flat = np.repeat(z_lens, n_td)
    z_src_flat = np.repeat(z_src, n_td)
    fpd_err_flat = fpd_std.reshape(-1)
    td_true_flat = td_mean.reshape(-1)
    td_err_flat = td_std.reshape(-1)
    mst_true_flat = np.repeat(mst_mean, n_td)
    mst_err_flat = np.repeat(mst_std, n_td)

    lens_id = np.repeat(np.arange(n_lens, dtype=int), n_td)
    pair_id = np.tile(np.arange(n_td, dtype=int), n_lens)
    block_id_arr = np.full(n_lens * n_td, block_id, dtype=int)

    return {
        "z_lens": z_lens_flat,
        "z_src": z_src_flat,
        "fpd_err": fpd_err_flat,
        "td_true": td_true_flat,
        "td_err": td_err_flat,
        "mst_true": mst_true_flat,
        "mst_err": mst_err_flat,
        "block_id": block_id_arr,
        "lens_id": lens_id,
        "pair_id": pair_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare quasar time-delay datavectors for inference.")
    parser.add_argument("--input", default="../Temp_data/static_datavectors_seed6.json", help="Input JSON file")
    parser.add_argument(
        "--output",
        default="/users/tianli/Temp_data/quasar_datavectors_seed6_processed.npz",
        help="Output NPZ file",
    )
    parser.add_argument("--min-err", type=float, default=1e-6, help="Floor for zero std errors")
    parser.add_argument(
        "--require-mst",
        action="store_true",
        help="Drop entries without MST (sigma_v/kin_pred) measurements",
    )
    args = parser.parse_args()

    infile = Path(args.input)
    with infile.open("r") as f:
        data = json.load(f)

    flat_accum: Dict[str, List[np.ndarray]] = {
        "z_lens": [],
        "z_src": [],
        "fpd_err": [],
        "td_true": [],
        "td_err": [],
        "mst_true": [],
        "mst_err": [],
        "block_id": [],
        "lens_id": [],
        "pair_id": [],
    }

    n_blocks = len(data)
    missing_mst_blocks = []

    for b, block in enumerate(data):
        z_lens = np.asarray(block["z_lens"], dtype=float)
        z_src = np.asarray(block["z_src"], dtype=float)

        fpd_samples = np.asarray(block["fpd_samples"], dtype=float)
        td_samples = np.asarray(block["td_measured"], dtype=float)

        _, fpd_std = _mean_std(fpd_samples, axis=1, min_err=args.min_err)
        td_mean, td_std = _mean_std(td_samples, axis=1, min_err=args.min_err)

        has_mst = ("sigma_v_measured" in block) and ("kin_pred_samples" in block)
        mst_mean, mst_std = _compute_mst_stats(block, min_err=args.min_err)
        if mst_mean is None or mst_std is None:
            missing_mst_blocks.append(b)
            if args.require_mst:
                continue
            mst_mean = np.full(z_lens.shape, np.nan)
            mst_std = np.full(z_lens.shape, np.nan)

        flat_block = _flatten_block(
            z_lens=z_lens,
            z_src=z_src,
            fpd_std=fpd_std,
            td_mean=td_mean,
            td_std=td_std,
            mst_mean=mst_mean,
            mst_std=mst_std,
            block_id=b,
        )

        for k in flat_accum:
            flat_accum[k].append(flat_block[k])

        print(
            f"Block {b:02d}: N_lens={td_mean.shape[0]}, N_td={td_mean.shape[1]}, MST={'yes' if has_mst else 'no'}"
        )

    # Concatenate
    out: Dict[str, np.ndarray] = {}
    for k, parts in flat_accum.items():
        out[k] = np.concatenate(parts, axis=0) if parts else np.array([])

    out["n_blocks"] = np.array([n_blocks], dtype=int)

    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outfile, **out)

    n_obs = out["z_lens"].shape[0]
    n_mst_ok = np.sum(np.isfinite(out["mst_true"]) & np.isfinite(out["mst_err"]))

    print("\nSummary:")
    print(f"  Input blocks: {n_blocks}")
    print(f"  Missing MST blocks: {missing_mst_blocks}")
    print(f"  Total observations (time delays): {n_obs}")
    print(f"  Observations with MST: {n_mst_ok}")
    print(f"  Saved: {outfile.resolve()}")


if __name__ == "__main__":
    main()
