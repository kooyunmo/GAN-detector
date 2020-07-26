import math
import time
from collections import OrderedDict
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchsummary import summary

from models.models import model_selection
from utils.args import parse_args

torch.backends.cudnn.benchmark = True


def preprocess(trainset_path, testset_path, classes=None, num_workers=4, batch_size=32, validation_ratio=0.2):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([299, 299]),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(trainset_path, transform=transform)
    test_dataset = datasets.ImageFolder(testset_path, transform=transform)

    num_trainset = len(train_dataset)
    indices = list(range(num_trainset))
    np.random.shuffle(indices)
    split = int(np.floor(validation_ratio * num_trainset))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=20, num_workers=num_workers, shuffle=True)

    print("Classes: ", classes)

    return train_loader, valid_loader, test_loader

def train(epochs, model, train_loader, valid_loader, criterion, optimizer):
    valid_loss_min = np.Inf
    train_vis = []
    valid_vis = []

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for step, (inp, target) in enumerate(train_loader):
            inp, target = inp.cuda(), target.cuda()

            optimizer.zero_grad()
            preds = model(inp)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inp.size(0)

            if step % 50 == 49:
                print("[EPOCH {}/{} BATCH {}/{}]\tloss: {}".format(epoch+1, epochs, step+1,
                                                                  len(train_loader.dataset),
                                                                  train_loss / (step+1)))

        model.eval()
        for inp, target in valid_loader:
            inp, target = inp.cuda(), target.cuda()

            preds = model(inp)
            loss = criterion(preds, target)
            valid_loss += loss.item() * inp.size(0)

        
        # Calculate average loss
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        train_vis.append(train_loss)
        valid_vis.append(valid_vis)

        print("#"*10 + " end of epoch " + "#"*10)
        print("[EPOCH {}] \ttrain loss: {:.6f} \tvalid_loss: {:.6f}".format(
            epoch+1, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            now = time.gmtime(time.time())
            time_info = str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
            torch.save(model.state_dict(), 'xception_gan-detector' + time_info + '.pt')
            valid_loss_min = valid_loss


if __name__ == '__main__':
    args = parse_args()

    train_loader, valid_loader, test_loader = preprocess(trainset_path=args.data_dir + "/train",
                                                         testset_path=args.data_dir + "/test",
                                                         classes=['stylegan', 'pggan', 'msgstylegan', 'vgan', 'real'],
                                                         num_workers=40,
                                                         batch_size=args.batch_size,
                                                         validation_ratio=0.3)
    
    model, image_size, *_ = model_selection('xception', num_out_classes=4)
    print(model)
    model = model.cuda()
    input_size = (3, image_size, image_size)
    print(summary(model, input_size))

    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=list(model.parameters())[:-1])

    train(args.epochs, model, train_loader, valid_loader, criterion, optimizer)
