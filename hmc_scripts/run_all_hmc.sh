#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO_ROOT"

python -u "$SCRIPT_DIR/run_lens_kin_hmc.py"
python -u "$SCRIPT_DIR/run_dspl_hmc.py"
python -u "$SCRIPT_DIR/run_sne_hmc.py"
python -u "$SCRIPT_DIR/run_quasar_hmc.py"
python -u "$SCRIPT_DIR/run_joint_hmc.py"
