# TensorFlow Implementation of PGGAN

## 0. Reference
- Progressive Growing of GANs for Improved Quality, Stability, and Variation [paper](https://arxiv.org/pdf/1710.10196.pdf)
- Official TensorFlow Implementations of PGGAN [link](https://github.com/tkarras/progressive_growing_of_gans)

## Purpose
- Fix minor codes of official tensorflow implementations of PGGAN on below environment

```
numpy == 1.18.5
scipy == 1.4.1
tensorflow-gpu == 2.2.0
moviepy == 0.2.3.5
Pillow == 7.0.0
lmdb == 0.98
opencv-python == 4.1.2.30
cryptography == 2.9.2
h5py == 2.10.0
six == 1.12.0
```

- Make fake images using pre-trained PGGAN on CelebA-HQ (1024*1024)
  * Process
  * 1. Download pre-trained and place in same directory as the script : [`karras2018iclr-celebahq-1024x1024.pkl`](https://drive.google.com/open?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4)
  * 2. Run : `Implementation.ipynb`
