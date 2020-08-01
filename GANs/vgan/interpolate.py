import argparse
import os
from os import path
import copy
import numpy as np
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist, interpolate_sphere
from gan_training.config import (load_config, build_models)

# Arguments
parser = argparse.ArgumentParser(
    description='Create interpolations for a trained GAN.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
interp_dir = path.join(out_dir, 'test', 'interp')

# Creat missing directories
if not path.exists(interp_dir):
    os.makedirs(interp_dir)

# Logger
checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

device = torch.device("cuda:0" if is_cuda else "cpu")

generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(
    config['z_dist']['type'], config['z_dist']['dim'], device=device)

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')

print('Creating interplations...')
nsteps = config['interpolations']['nzs']
nsubsteps = config['interpolations']['nsubsteps']

y = ydist.sample((sample_size, ))
zs = [zdist.sample((sample_size, )) for i in range(nsteps)]
ts = np.linspace(0, 1, nsubsteps)

for i in range(10):
    zs = [zdist.sample((sample_size, )) for i in range(nsteps)]
    it = 0
    for z1, z2 in zip(zs, zs[1:] + [zs[0]]):
        for t in ts:
            z = interpolate_sphere(z1, z2, float(t))
            with torch.no_grad():
                x = generator_test(z, y)
                out_path = path.join(interp_dir, '%03d_%04d.png' % (i, it))
                utils.save_images(x, out_path, nrow=sample_nrow)

                it += 1
                print('%d: %d/%d done!' % (i, it, nsteps * nsubsteps))
