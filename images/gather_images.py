import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True,
                        help='root directory that contains images to gather')
    parser.add_argument('--output-name', type=str, required=True,
                        help='output file name')
    return parser.parse_args()


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


if __name__ == '__main__':
    args = get_args()
    path = os.path.join(args.root_dir, '*.png')
    images = [Image.open(x) for x in glob.glob(path)]

    new_im = pil_grid(images, max_horiz=5)

    new_im.save(os.path.join(args.root_dir, args.output_name))
