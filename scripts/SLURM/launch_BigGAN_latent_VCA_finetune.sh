#!/bin/bash
#SBATCH --job-name=vca-gan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bjohn.braddock@ufl.edu

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=5gb


#SBATCH --time=2:00:00
#SBATCH --output=logs/SLURM/BigGAN_finetune_%j.log

pwd; hostname; date

module load conda

conda activate emoteGAN



python /home/bjohn.braddock/BigGAN-PyTorch/finetune_latent_with_vca.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 4 --batch_size 1   \
--num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-3 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--use_multiepoch_sampler \
--resume --load_weights_root "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained/138k" \
--weights_root "/blue/ruogu.fang/bjohn.braddock/BigGAN/savedmodels/BigGAN"  \
--samples_root "/blue/ruogu.fang/bjohn.braddock/BigGAN/samples" \
--vca_filepath "/blue/ruogu.fang/bjohn.braddock/BigGAN/pretrained/VCA/best_model_emotion_regression_amygdala_100epoch_model2_0117_12PM_epoch17.pth" \
--num_epochs 25 --iters_per_epoch 1000 --num_G_accumulations 8 \
--test_every 2000 --save_every 2000 --num_best_copies 5 --num_save_copies 0 --seed 0 \
--truncated_z 

date