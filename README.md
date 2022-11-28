# High-Fidelity GAN Inversion for Natural Image Attribute Editing by Removing Image Ripples

[사진 재현 및 편집 기술 연구] 임에딘 (지도 교수님 : 박경문)

## Overview
 > 최근 GAN Inversion을 기반으로 사진 합성 기술이 연구되고 있다. 이미지의 재구성과 편집 사이에는 Trade-off 문제가 있으며, 최신의 GAN Inversion 프레임워크인 High-Fidelity GAN Inversion (HFGI, CVPR 2022)는 재구성과 편집에서 균형을 맞춰 극복하였지만 시점 변경이 큰 이미지를 편집할 때 원본의 잔상(잔물결)이 남는 치명적인 한계를 보였다.

#### 지도교수님
- 박경문

#### 발표자
- 2017104019 임에딘

---
## 연구 배경

## Usage
- DataSet (Train: FFHQ, Eval: CelebA-HQ)   
  https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0  （contain preprocessed data)   
    받은 Dataset은 experiments/ecg/dataset/preprocessed/에 넣는다.   
- Inference
    `python ./scripts/inference.py \
--images_dir=./test_imgs  --n_sample= { } --edit_attribute='pose' --edit_degree=3  \
--save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt  \
--loadmodel=./pretrained/inference.pth  --output_path=./experiment/inference_results/pose_mask  \
--output_path2=./experiment/inference_results/original_mask`

    `cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/inversion2 outdir=$(pwd)/output`


## Environment
- OS : Ubuntu 20.04.5 LTS (GNU/Linux 5.4.0-131-generic x86_64) 
- VGA : NVIDIA GeForce RTX 3090 
- CPU : Intel(R) Xeon(R) E-2334 CPU @ 3.40GHz 
- python 3.7.13
- PyTorch 1.7.1
- cuda 11.0

    `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html`

## Conclusion
> HFGI에서는 시점 변경이 크게 속성 편집을 한 이미지에는 잔상이 남는 문제점이 있었다. 이를 해결하기 위해 Graphonomy를 이용하여 각 이미지들의 이진 마스크를 생성하고 잔물결에 해당하는 부분을 구했다. 시점 변경이 되면서 기존의 인물이 없는 빈 공간 (ghost artifact)의 경우 image inpainting 기법을 이용하여 누락된 부분을 채우고 새롭게 합성하는 방식을 취하였다.


## Reference
[HFGI](https://github.com/Tengfei-Wang/HFGI)   
[Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy)
[LaMa](https://github.com/saic-mdal/lama)   

## Reports
