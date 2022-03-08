#!/bin/bash
#SBATCH --job-name=vca-gan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bjohn.braddock@ufl.edu

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=10gb

#SBATCH  --account=ruogu.fang
#SBATCH  --qos=ruogu.fang

#SBATCH --time=1:10:00
#SBATCH --output=convert_from_tfhub_%j.log

pwd; hostname; date

module load conda

conda activate emoteGAN



python /home/bjohn.braddock/BigGAN-PyTorch/converter.py \
-r 256 --weights_dir "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained/"

date