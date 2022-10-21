#!/bin/bash

#SBATCH --job-name train_hfgi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 3-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -x sw1
#SBATCH -o slurm/logs/slurm-%A-%x.out

python ./scripts/train.py   --dataset_type='ffhq_encode'  --start_from_latent_avg \
--id_lambda=0.1  --val_interval=20000 --save_interval=20000 --max_steps=100000  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0.1  \
--stylegan_weights='./pretrained/stylegan2-ffhq-config-f.pt' --checkpoint_path='./pretrained/e4e_ffhq_encode.pt'  \
--workers=48  --batch_size=8  --test_batch_size=4 --test_workers=48 --exp_dir='./experiment/ffhq' 
