#!/bin/sh
#SBATCH -J HOPS_trunc_16
#SBATCH -o ./N.out
#SBATCH -D ./
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mem=10mb
#SBATCH --mail-user=benjamin.sappler@tum.de
#SBATCH --export=NONE
#SBATCH --time=02:00:00
module load slurm_setup
module load python
source ../../../HOPS_env/bin/activate
python script.py
source ../../../HOPS_env/bin/deactivate
