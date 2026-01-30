#!/usr/bin/env python3
"""
Dump full structure of static_datavectors JSON:
types, keys, and array-like shapes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _is_numeric_array_list(x) -> bool:
    if not isinstance(x, list):
        return False
    arr = np.asarray(x)
    return arr.dtype != object


def _describe_list(x) -> str:
    arr = np.asarray(x)
    return f"list->array shape={arr.shape}, dtype={arr.dtype}"


def walk(obj, prefix: str) -> None:
    if isinstance(obj, dict):
        print(f"{prefix} (dict, {len(obj)} keys)")
        for k in obj:
            walk(obj[k], f"{prefix}.{k}")
        return

    if isinstance(obj, list):
        if _is_numeric_array_list(obj):
            print(f"{prefix} ({_describe_list(obj)})")
            return
        print(f"{prefix} (list, len={len(obj)})")
        for i, v in enumerate(obj):
            walk(v, f"{prefix}[{i}]")
        return

    print(f"{prefix}: {type(obj).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump full structure of static_datavectors JSON.")
    parser.add_argument("--input", default="../Temp_data/static_datavectors_seed6.json", help="Input JSON file")
    args = parser.parse_args()

    with Path(args.input).open("r") as f:
        data = json.load(f)

    top_type = type(data).__name__
    if isinstance(data, list):
        print(f"Top-level: list, len={len(data)}")
    elif isinstance(data, dict):
        print(f"Top-level: dict, keys={len(data)}")
    else:
        print(f"Top-level: {top_type}")

    walk(data, "root")


if __name__ == "__main__":
    main()
