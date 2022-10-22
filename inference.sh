#!/bin/bash

#SBATCH --job-name inference_hfgi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 3-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -x sw1
#SBATCH -o slurm/logs/slurm-%A-%x.out

#python ./scripts/inference.py \
#--images_dir=./test_imgs  --n_sample=76 --edit_attribute='inversion'  \
#--save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

python ./scripts/inference.py \
--images_dir=./test_imgs  --n_sample=76 --edit_attribute='pose' --edit_degree=3  \
--save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=76 --edit_attribute='lip'  \
# --save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=76 --edit_attribute='beard'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=76 --edit_attribute='eyes'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

#python ./scripts/inference.py \
#--images_dir=./test_imgs  --n_sample=76 --edit_attribute='smile' --edit_degree=1.0  \
#--save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=76 --edit_attribute='age' --edit_degree=3  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 
