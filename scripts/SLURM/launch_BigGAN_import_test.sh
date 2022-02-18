#!/bin/bash
#SBATCH --job-name=slurm_job_test 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bjohn.braddock@ufl.edu

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=8gb


#SBATCH --time=00:30:00
#SBATCH --output=BigGAN_test_%j.log

pwd; hostname; date

module load conda

conda activate emoteGAN



python /home/bjohn.braddock/BigGAN-PyTorch/import_test.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 16   \
--num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--use_multiepoch_sampler \
--resume --weights_root "./models/BigGAN" --experiment_name "138k" \
--vca_filepath "./pretrained/VCA/model1_test_epoch50.pth" \
--num_epochs 10 --iters_per_epoch 1000 --num_G_accumulations 8 \
--test_every 2000 --save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0

date