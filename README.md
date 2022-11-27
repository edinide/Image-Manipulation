# Image-Manipulation
[사진 재현 및 편집 기술 연구] 임에딘 (지도 교수님 : 박경문)

* Environment
- OS : Ubuntu 20.04.5 LTS (GNU/Linux 5.4.0-131-generic x86_64) 
- VGA : NVIDIA GeForce RTX 3090 
- CPU : Intel(R) Xeon(R) E-2334 CPU @ 3.40GHz 


* inference
python ./scripts/inference.py \
--images_dir=./test_imgs  --n_sample= { } --edit_attribute='pose' --edit_degree=3  \
--save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt  \
--loadmodel=./pretrained/inference.pth  --output_path=./experiment/inference_results/pose_mask  \
--output_path2=./experiment/inference_results/original_mask

cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/inversion2 outdir=$(pwd)/output

* HFGI : https://github.com/Tengfei-Wang/HFGI
* lama : https://github.com/saic-mdal/lama
