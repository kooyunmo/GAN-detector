import os
import shutil
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2

from models.models import model_selection
from utils.preprocess import preprocess
from utils.plot import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='model phase: choose between `train` and `test`.')
    parser.add_argument('--data-dir', type=str, default='./datasets', help='root of data directory')
    parser.add_argument('--model-name', type=str, default='xception', help='model name (e.g. xception, resnet101, etc)')
    parser.add_argument('--model-path', type=str, help='model checkpoint path')
    parser.add_argument('--num-epochs', type=int, default=30, help='the number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='training batch size')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='directory to save model checkpoints')
    parser.add_argument('--gpu', type=str, default='0', help='gpu pci id')
    return parser.parse_args()


def imsave(inp, filename):
    inp = inp[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave(filename, inp)


def train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs=30):
    start = time.time()
    train_loss_log, valid_loss_log, train_acc_log, valid_acc_log, epoch_log = list(), list(), list(), list(), list()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("=" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)    # return predicted class indices
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print("{} Loss: {:.4f}\t Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'train':
                train_loss_log.append(epoch_loss)
                train_acc_log.append(epoch_acc.item())
            else:
                valid_loss_log.append(epoch_loss)
                valid_acc_log.append(epoch_acc.item())
        epoch_log.append(epoch)
        visualize(model_name, train_loss_log, valid_loss_log, train_acc_log, valid_acc_log, epoch_log)
        print()
    
    elapsed_time = time.time() - start
    print("Training complete in {:.0f}m {:.0f}s".format(
        elapsed_time // 60, elapsed_time % 60))
    print("Best validation acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloaders):
    raise NotImplementedError


def visualize_model(model, dataloaders, classes, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(classes[preds[j]]))
                imsave('pred_examples.png', inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def main():
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Configuration: ", args)

    dataloaders = preprocess(args.model_name, args.data_dir, 4, args.batch_size, 0.2)

    # inputs, classes = next(iter(dataloaders['val']))
    # out = torchvision.utils.make_grid(inputs)
    # imsave(inputs, 'example_image.png')

    if args.model_path is not None:
        print("Load model from '{}'".format(args.model_path))
        model = torch.load(args.model_path)
    else:
        model, *_ = model_selection(args.model_name, len(dataloaders['train'].dataset.classes))
    model = model.cuda()
    hw_size = 224 if 'resnet' in args.model_name else 299
    print(summary(model, (3, hw_size, hw_size)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if args.phase == 'train':
        model = train_model(model=model,
                            model_name=args.model_name,
                            dataloaders=dataloaders,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=exp_lr_scheduler,
                            num_epochs=args.num_epochs)
        torch.save(model, os.path.join(args.save_dir, 'gan-detection-' + args.model_name + '.h5'))
    else:
        test_model(model, dataloaders)


if __name__ == '__main__':
    main()