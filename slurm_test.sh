#!/bin/bash
#SBATCH --job-name=slurm_job_test 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bjohn.braddock@ufl.edu

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=500mb


#SBATCH --time=00:05:00
#SBATCH --output=serial_test_%j.log

pwd; hostname; date

module load conda

conda activate emoteGAN

echo "Running test script in conda env.."

python /home/bjohn.braddock/BigGAN-PyTorch/test.py

date
