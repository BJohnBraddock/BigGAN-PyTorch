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
#SBATCH --output=logs/SLURM/BigGAN_finetune_%j.log

pwd; hostname; date

module load conda

conda activate emoteGAN



python /home/bjohn.braddock/BigGAN-PyTorch/finetune_class_with_vca.py \
--seed 123850 --lr 1e-4 --batch_size 1 --truncation 0.5 \
--num_epochs 30 --iters_per_epoch 1000 \
--log_every 200 \
--vca_filepath "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained/VCA/best_model_emotion_regression_amygdala_100epoch_model2_0117_12PM_epoch17.pth" \
--biggan_cache_dir "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained" 


date