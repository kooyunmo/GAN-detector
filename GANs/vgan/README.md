# Pytorch Implementation of VGAN

## 0. Reference
- Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow [paper](https://arxiv.org/pdf/1810.00821.pdf)
- Official Pytorch Implementations of VGAN [link](https://github.com/akanazawa/vgan)

## Purpose
- Fix minor codes of official pytorch implementations of VGAN on below environment

```
absl-py == 0.9.0
cycler == 0.10.0
decorator == 4.4.2
imageio == 2.4.1
ipdb == 0.13.3
ipython == 5.5.0
ipython-genutils == 0.2.0
kiwisolver == 1.2.0
matplotlib == 3.2.2
numpy == 1.18.5
opencv-python == 4.1.2.30
pexpect == 4.8.0
pickleshare == 0.7.5
Pillow == 7.0.0
protobuf == 3.12.2
ptyprocess == 0.6.0
pyparsing == 2.4.7
python-dateutil == 2.8.1
pytz == 2018.9
PyYAML == 3.13
scipy == 1.1.0
simplegeneric == 0.8.1
six == 1.15.0
tensorboardX == 2.1
torchvision == 0.6.1 + cu101
traitlets == 4.3.3
```

- Make fake images using pre-trained VGAN on CelebA-HQ (1024*1024)
```
vgan
  ├ output
  |   ├ img
  |   |  └ generated images
  |   └ pretrained
  |      └ CelebA_HQ_vgan_model.pt
  |
  ├ Implementation.ipynb
  └ celebAHQ_vgan.1-gp.yaml

```
  * Process
  * 1. Download pre-trained and place in directory './output/pretrained' : [`CelebA_HQ_vgan_model.pt`](https://drive.google.com/file/d/1qwmQtkGm-i0IKFqSEVWugh4DVpJW4CQJ/view)
  * 2. Edit `celebAHQ_vgan.1-gp.yaml` test sample size for the image you want to generate
  * 3. Run : `Implementation.ipynb`
