#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

repo_root = Path(__file__).resolve().parent.parent
os.chdir(repo_root)

DATA_DIR = Path(os.environ.get("SLCOSMO_DATA_DIR", str(repo_root / "data")))
QUASAR_JSON = Path(os.environ.get("QUASAR_JSON", str(repo_root / "../Temp_data/static_datavectors_seed6.json")))

required = [
    DATA_DIR / "EuclidDSPLs_1.txt",
    DATA_DIR / "Euclid_len.txt",
    DATA_DIR / "Euclid_150SNe.csv",
    DATA_DIR / "velocity_disp_table.npy",
    QUASAR_JSON,
]

missing = [p for p in required if not p.exists()]

print("Data paths:")
for p in required:
    print(f"  - {p}")
if missing:
    print("Missing required inputs:")
    for p in missing:
        print(f"  - {p}")
    raise SystemExit(1)

scripts = [
    "hmc_scripts/run_lens_kin_hmc.py",
    "hmc_scripts/run_dspl_hmc.py",
    "hmc_scripts/run_sne_hmc.py",
    "hmc_scripts/run_quasar_hmc.py",
    "hmc_scripts/run_joint_hmc.py",
]

for script in scripts:
    subprocess.run(["python", "-u", script], check=True)
