#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=inspect_mst
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=log/inspect_mst_%j.out
#SBATCH --error=log/inspect_mst_%j.err
#SBATCH --mail-user=tian.li@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=sciama4.q

module load system
module load anaconda3/2024.02
echo `module list`

source activate gpu_herculens_2
cd /users/tianli/LensedUniverse

mkdir -p log

python -u inspect_mst_sources.py --input ../Temp_data/static_datavectors_seed6.json
