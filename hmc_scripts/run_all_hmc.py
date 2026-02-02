#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

repo_root = Path(__file__).resolve().parent.parent
os.chdir(repo_root)

scripts = [
    "hmc_scripts/run_lens_kin_hmc.py",
    "hmc_scripts/run_dspl_hmc.py",
    "hmc_scripts/run_sne_hmc.py",
    "hmc_scripts/run_quasar_hmc.py",
    "hmc_scripts/run_joint_hmc.py",
]

for script in scripts:
    subprocess.run(["python", "-u", script], check=True)
