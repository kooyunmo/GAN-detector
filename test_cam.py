#!/usr/bin/env python
# coding: utf-8
# Modified from original repo: https://github.com/kazuto1011/grad-cam-pytorch


from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
from PIL import Image
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from utils.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths, arch):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path, arch)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = ['msgstylegan', 'pggan', 'stylegan', 'vgan']
    return classes



def preprocess(image_path, arch):
    transform_size = 224 if 'resnet' in arch else 299
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (transform_size,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


def get_model(arch):
    model = torch.load('checkpoints/gan-detection-' + arch + ".h5")
    return model


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.append('xception')

@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=4)
@click.option("-o", "--output-dir", type=str, default="./cam_results/demo1")
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()
    print("classes: ", classes)

    # Model from torchvision
    #model = models.__dict__[arch](pretrained=True)
    model = get_model(arch)
    model.to(device)
    model.eval()
    print(model)

    # Images
    images, raw_images = load_images(image_paths, arch)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted

    for i in range(topk):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "top{}-{}-vanilla-{}.png".format(i, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "top{}-{}-deconvnet-{}.png".format(i, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "top{}-{}-guided-{}.png".format(i, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "top{}-{}-gradcam-{}-{}.png".format(
                        i, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "top{}-{}-guided_gradcam-{}-{}.png".format(
                        i, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-o", "--output-dir", type=str, default="./cam_results/demo2")
@click.option("-c", "--trgt_class", type=str, default=None)
@click.option("--cuda/--cpu", default=True)
def demo2(image_paths, arch, trgt_class, output_dir, cuda):
    """
    Generate Grad-CAM at different layers
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()
    print("classes: ", classes)

    # Model
    print("Grad-cam with ", arch)
    model = get_model(arch)
    model.to(device)
    model.eval()

    # The four residual layers
    if arch == 'xception':
        target_layers = ['model.block1', 'model.block2', 'model.block3', 'model.block4',
                         'model.block5', 'model.block6', 'model.block7', 'model.block8',
                         'model.block9', 'model.block10', 'model.block11', 'model.block12']
    elif arch == 'resnet101' or arch == 'resnet50':
        target_layers = ["model.relu", "model.layer1", "model.layer2", "model.layer3", "model.layer4"]
    else:
        raise NotImplementedError

    if trgt_class is None:
        target_class = classes.index(image_paths[0].split('/')[2])
    else:
        target_class = classes.index(trgt_class)

    # Images
    images, raw_images = load_images(image_paths, arch)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-gradcam-{}-{}.png".format(
                        arch, target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=4)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./cam_results")
@click.option("--cuda/--cpu", default=True)
def demo3(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    #model = models.__dict__[arch](pretrained=True)
    model = get_model(arch)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


if __name__ == "__main__":
    main()
