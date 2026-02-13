#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=jaxlcosmo
#SBATCH --ntasks-per-node=20
#SBATCH --time=24:00:00
#SBATCH --output=log/output_log%j
#SBATCH --error=log/error_log%j
#SBATCH --mail-user=tian.li@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu.q
#SBATCH --exclude gpu0[1]

module load system
module load anaconda3/2024.02
echo `module list`

source /mnt/lustre2/shared_conda/envs/tianli/herculens_tian/bin/activate
cd /users/tianli/LensedUniverse

export SLCOSMO_USE_X64=1
export SLCOSMO_RUN_NOISY=0

echo "[STEP] Run joint HMC"
python -u hmc_scripts/run_joint_hmc.py

echo "[STEP] Run lens+kinematic HMC"
python -u hmc_scripts/run_lens_kin_hmc.py

echo "[STEP] Run DSPL HMC"
python -u hmc_scripts/run_dspl_hmc.py

echo "[STEP] Run lensed SNe HMC"
python -u hmc_scripts/run_sne_hmc.py

echo "[STEP] Run lensed quasar HMC"
python -u hmc_scripts/run_quasar_hmc.py
