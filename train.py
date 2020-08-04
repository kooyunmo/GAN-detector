import time
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchsummary import summary

from models.models import model_selection
from utils.args import parse_args
from utils.preprocess import preprocess
from utils.plot import visualize

torch.backends.cudnn.benchmark = True


def train(epochs, model, train_loader, valid_loader, criterion, optimizer, result_dir):
    valid_loss_min = np.Inf
    train_vis = []
    valid_vis = []
    epoch_vis = []

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # Training
        model.train()
        for _, (inp, target) in enumerate(tqdm(train_loader)):
            inp, target = inp.cuda(), target.cuda()

            optimizer.zero_grad()
            preds = model(inp)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inp.size(0)
            
        # Validating
        model.eval()
        for inp, target in tqdm(valid_loader):
            inp, target = inp.cuda(), target.cuda()

            preds = model(inp)
            loss = criterion(preds, target)
            valid_loss += loss.item() * inp.size(0)

        
        # Calculate average loss
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        train_vis.append(train_loss)
        valid_vis.append(valid_loss)

        print("#"*10 + " end of epoch " + "#"*10)
        print("[EPOCH {}] \ttrain loss: {:.6f} \tvalid_loss: {:.6f}".format(
            epoch+1, train_loss, valid_loss))

        # Save the best model checkpoint
        if valid_loss <= valid_loss_min:
            print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            # now = time.gmtime(time.time())
            # time_info = str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
            torch.save(model.state_dict(), os.path.join(result_dir, 'gan-detector-xception.pt'))
            valid_loss_min = valid_loss

        epoch_vis.append(epoch)
        visualize(train_vis, valid_vis, epoch_vis, result_dir)


if __name__ == '__main__':
    args = parse_args()

    train_loader, valid_loader, test_loader = preprocess(trainset_path=args.data_dir + "/train",
                                                         testset_path=args.data_dir + "/test",
                                                         num_workers=40,
                                                         batch_size=args.batch_size,
                                                         validation_ratio=0.3)
    
    model, image_size, *_ = model_selection('xception', num_out_classes=4)
    print(model)
    model = model.cuda()
    input_size = (3, image_size, image_size)
    print(summary(model, input_size))

    #model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=list(model.parameters())[:-1])

    train(args.epochs, model, train_loader, valid_loader, criterion, optimizer, args.result_dir)
