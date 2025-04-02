## Introduction
This is a repository about the paper "ChangeRD: A registration-integrated change detection framework for unaligned remote sensing images"[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0924271624004635).

## :speech_balloon: Requirements

```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2

Please see `requirements.txt` for all the other requirements.

```



## :speech_balloon: Train on LEVIR-CD

When we initialy train our ChangeRD, we initialized some parameters of the network with a model pre-trained on the RGB segmentation (ADE 160k dataset) to get faster convergence.

You can download the pre-trained model [`here`](https://pan.baidu.com/s/1YULSIq2_VlfdnZ1FNTtZSw). PASSWORD: qm5e

Then, update the path to the pre-trained model by updating the ``path`` argument in the ``run_ChangeRD_LEVIR.sh``.

You can find the training script `run_ChangeRD_LEVIR.sh` in the folder `scripts`. You can run the script file by `sh scripts/run_ChangeRD_LEVIR.sh` in the command environment.


## :speech_balloon: Evaluate on LEVIR

You can find the evaluation script `eval_ChangeRD_LEVIR.sh` in the folder `scripts`. You can run the script file by `sh scripts/eval_ChangeRD_LEVIR.sh` in the command environment.

Note: During testing, you can run `gen_offset_data.py` to manually warp the image to be tested to generate unaligned image pairs

## :speech_balloon: Dataset Preparation

### :point_right: Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

## :speech_balloon: License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## :speech_balloon: Citation

If you use this code for your research, please cite our paper:

```
@article{JING202564,
title = {ChangeRD: A registration-integrated change detection framework for unaligned remote sensing images},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {220},
pages = {64-74},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.11.019},
url = {https://www.sciencedirect.com/science/article/pii/S0924271624004635},
author = {Wei Jing and Kaichen Chi and Qiang Li and Qi Wang},
keywords = {Change detection, Registration, Remote sensing, Deep learning, Perspective transformation},
abstract = {Change Detection (CD) is important for natural disaster assessment, urban construction management, ecological monitoring, etc. Nevertheless, the CD models based on the pixel-level classification are highly dependent on the registration accuracy of bi-temporal images. Besides, differences in factors such as imaging sensors and season often result in pseudo-changes in CD maps. To tackle these challenges, we establish a registration-integrated change detection framework called ChangeRD, which can explore spatial transformation relationships between pairs of unaligned images. Specifically, ChangeRD is designed as a multitask network that supervises the learning of the perspective transformation matrix and difference regions between images. The proposed Adaptive Perspective Transformation (APT) module is utilized to enhance spatial consistency of features from different levels of the Siamese network. Furthermore, an Attention-guided Central Difference Convolution (AgCDC) module is proposed to mine the deep differences in bi-temporal features, significantly reducing the pseudo-change noise caused by illumination variations. Extensive experiments on unaligned bi-temporal images have demonstrated that ChangeRD outperforms other SOTA CD methods in terms of qualitative and quantitative evaluation. The code for this work will be available on GitHub.}
}
```

## :speech_balloon: References
Appreciate the work from the following repositories:

- https://github.com/wgcban/ChangeFormer (Our ChangeRD is implemented on the code provided in this repository)

