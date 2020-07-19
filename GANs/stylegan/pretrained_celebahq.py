# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# Modified for generating CelebA HQ fake images

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import sys
import argparse
import random

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-num-start', type=int, default=0, help='A start image number which will be used as a file name.')
    parser.add_argument('--num-imgs', type=int, help='The number of images you want to generate')
    parser.add_argument('--result-dir', type=str, help='Directory path to save generated imgs')
    parser.add_argument('--gpu-num', type=int, default=0, help='gpu device number')
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

    # Command line authentication to your Google Drive
    # gauth = GoogleAuth()
    # gauth.LocalWebserverAuth()
    # drive = GoogleDrive(gauth)


    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    filename = "karras2019stylegan-celebahq-1024x1024.pkl"
    with open(filename, "rb") as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    rnd = np.random.RandomState(random.randint(1, 1000))
    os.makedirs(args.result_dir, exist_ok=True) # C:/stylegan/result
    for i in range(args.img_num_start, args.img_num_start + args.num_imgs):
        # Pick latent vector.
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        zero_len = len(str(args.num_imgs)) - len(str(i))
        png_filename = os.path.join(args.result_dir, '0'*zero_len + str(i) + ".png")
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

        # f = drive.CreateFile({'title': i + ".png"})
        # f.SetContentFile(i + ".png")
        # f.Upload()
        # f = None        # for preventing memory leak

if __name__ == "__main__":
    main()
