#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=slcosmo
#SBATCH --time=120:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=log/output_log%j
#SBATCH --error=log/error_log%j
#SBATCH --mail-user=tian.li@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu.q
#SBATCH --exclude gpu[01,02]

module load system
module load anaconda3/2024.02
echo `module list`

source /mnt/lustre2/shared_conda/envs/tianli/herculens_tian/bin/activate
cd /users/tianli/LensedUniverse

python -u hmc_scripts/run_lens_kin_hmc.py
python -u hmc_scripts/run_dspl_hmc.py
python -u hmc_scripts/run_sne_hmc.py
python -u hmc_scripts/run_quasar_hmc.py
python -u hmc_scripts/run_joint_hmc.py
