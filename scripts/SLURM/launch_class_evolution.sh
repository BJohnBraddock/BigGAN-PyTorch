#!/bin/bash
#SBATCH --job-name=vca-gan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bjohn.braddock@ufl.edu

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=5gb


#SBATCH --time=1:00:00
#SBATCH --output=logs/SLURM/evolution_%j.log

pwd; hostname; date

module load conda

conda activate emoteGAN


python /home/bjohn.braddock/BigGAN-PyTorch/genetic_algorithm_GAN.py \
--seed 4321 --population_size 128 --truncation 0.5 \
--mutate_probability 0.4 \
--generation_limit 1000 --fitness_limit 0.95 \
--log_every 10 --image_log_size 4 \
--vca_filepath "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained/VCA/best_model_emotion_regression_amygdala_100epoch_model2_0117_12PM_epoch17.pth" \
--biggan_cache_dir "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained" 


date