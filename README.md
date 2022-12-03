# High-Fidelity GAN Inversion for Natural Image Attribute Editing by Removing Image Ripples

[사진 재현 및 편집 기술 연구] 임에딘 (지도 교수님 : 박경문)

## Overview
 > 최근 GAN Inversion을 기반으로 사진 합성 기술이 연구되고 있다. 이미지의 재구성과 편집 사이에는 Trade-off 문제가 있으며, 최신의 GAN Inversion 프레임워크인 High-Fidelity GAN Inversion (HFGI, CVPR 2022)는 재구성과 편집에서 균형을 맞춰 극복하였지만 시점 변경이 큰 이미지를 편집할 때 원본의 잔상(잔물결)이 남는 치명적인 한계를 보였다.
---
## 연구 배경
최근 GAN Inversion을 기반으로 사진 합성 기술이 연구되고 있다. 이미지의 재구성과 편집 사이에는 Trade-off 문제가 있으며, 최신의 GAN Inversion 프레임워크인 High-Fidelity GAN Inversion (HFGI, CVPR 2022)는 재구성과 편집에서 균형을 맞춰 극복하였지만 시점 변경이 큰 이미지를 편집할 때 잔상이 남는 치명적인 한계를 보였다. 따라서 HFGI에서 ‘pose’ 속성 편집을 했을 때 이미지의 잔상을 최대한 제거하여 자연스러운 이미지 편집이 가능하도록 개선하는 것이 본 연구의 목표이다.

## 주요 내용
제안하는 모델의 구조는 아래의 그림에 묘사되어 있다. HFGI에서 나온 이미지들을 사전훈련이 된 Graphonomy 모델을 통해 이진마스크 사진을 생성한 후, 비트연산을 통해서 이미지 잔상에 해당하는 부분에 대해 이진마스크를 생성하였다. 또한 시점 변경으로 인물이 움직이면서 원래는 없었던 배경이 새롭게 합성되어야 하는데, 이 누락된 부분을 더 좋게 채울 수 있도록 Image Inpainting 기법을 적용했다. Image inpainting은 이미지에서 누락된 부분을 재구성하는 기술로 현재까지 다양한 기법들이 등장하였으며, 그 중에서 올해 발표된 기법 중 256*256 해상도에서 훈련하고 고해상도의 이미지를 넓게 인페인팅이 가능한 성능 좋은 lama inpainting(WACV 2022)을 이용하였다. 제공된 pretrained model를 이용하여 편집된 이미지와 잔물결 이진 마스크 두개를 입력으로 넣어서 해당 부분을 지우고 자연스럽게 채워주는 lama inpainting을 적용했다.

![architecture](https://user-images.githubusercontent.com/30232133/205449282-a8783050-9793-48a3-a6b2-cc88a3a0130d.jpg)
---
## Environment
- OS : Ubuntu 20.04.5 LTS (GNU/Linux 5.4.0-131-generic x86_64) 
- VGA : NVIDIA GeForce RTX 3090 
- CPU : Intel(R) Xeon(R) E-2334 CPU @ 3.40GHz 
- python 3.7.13
- PyTorch 1.7.1
- cuda 11.0

## Set up
### Installation
```
git clone https://github.com/edinide/Image-Manipulation.git
cd HFGI
```

### Environment
```
conda create -n HFGI python=3.7
conda activate HFGI
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib
conda install ninja
conda install -c 3dhubs gcc-5
```

### Dataset
- Train : FFHQ, Eval : CelebA-HQ
- There are some images from CelebA-HQ in `./test_imgs`, and you can quickly try them (and other images from CelebA-HQ or FFHQ). 

### Inference
Modify `inference.sh` according to the follwing instructions, and run:   
```
bash inference.sh 
```
If you are running this code in slurm, `slurm : sbatch inference.sh`
Then you can get inpainted images by running lama, run:
```
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/inversion2 outdir=$(pwd)/output`
```

| Args | Description
| :--- | :----------
| --images_dir | the path of images.
| --n_sample | number of images that you want to infer.
| --edit_attribute | HFGI provides options of 'inversion', 'age', 'smile', 'eyes', 'lip' and 'beard' in the script. Also, I edited codes so that this project can provides option of 'pose'.
| --edit_degree | control the degree of editing (works for 'age', 'smile', and 'pose'. ex) 'pose' degree range is -3.0~ 3.0).
| --save_dir | the path of saved images.
| --loadmodel | the path of pretrained Graphonomy model.
| --output_path | the path of edited images' binary masks.
| --output_path2 | the path of original images' binary masks.

## Training (same with HFGI project)
### Preparation
1. Download datasets and modify the dataset path in `./configs/paths_config.py` accordingly.
2. Download some pretrained models and put them in `./pretrained`.

| Model | Description
| :--- | :----------
|[StyleGAN2 (FFHQ)](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | Pretrained face generator on FFHQ  from [rosinality](https://github.com/rosinality/stylegan2-pytorch).
|[e4e (FFHQ)](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing) | Pretrained initial encoder on FFHQ  from [omertov](https://github.com/omertov/encoder4editing).
|[Feature extractor (for face)](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for ID loss calculation.

### Start Training
Modify `option` and `training.sh` and run:
```
bash train.sh
```
If you are running this code in slurm, `slurm : sbatch train.sh`

## Conclusion
> HFGI에서는 시점 변경이 크게 속성 편집을 한 이미지에는 잔상이 남는 문제점이 있었다. 이를 해결하기 위해 Graphonomy를 이용하여 각 이미지들의 이진 마스크를 생성하고 잔물결에 해당하는 부분을 구했다. 시점 변경이 되면서 기존의 인물이 없는 빈 공간 (ghost artifact)의 경우 image inpainting 기법을 이용하여 누락된 부분을 채우고 새롭게 합성하는 방식을 취하였다.

## Reference
[HFGI](https://github.com/Tengfei-Wang/HFGI)   
[Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy)
[LaMa](https://github.com/saic-mdal/lama)   

## Reports
