import argparse
import random
import os
import warnings
import PIL.Image
import pickle

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default='Images',
                        help='output directory to save generated images')
    parser.add_argument('--num-gen-imgs', type=int, default=10,
                        help='the number of images to generate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Initialize TensorFlow session.
    sess=tf.InteractiveSession()

    # Import official CelebA-HQ networks.
    with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
        G, D, Gs = pickle.load(file)

    # Generate latent vectors.
    latents = np.random.RandomState(random.randint(1, 1000)).randn(args.num_gen_imgs, *Gs.input_shapes[0][1:]) # 1000 random latents

    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latents, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    # Save images as PNG.
    for idx in range(images.shape[0]):
        png_filename = os.path.join(args.result_dir, str(idx+1) + ".png")
        PIL.Image.fromarray(images[idx], 'RGB').save(png_filename)

if __name__ == '__main__':
    main()
