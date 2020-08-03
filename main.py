import os
import shutil
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./datasets', help='root of data directory')
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


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=30):
    start = time.time()

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
        
        print()
    
    elapsed_time = time.time() - start
    print("Training complete in {:.0f}m {:.0f}s".format(
        elapsed_time // 60, elapsed_time % 60))
    print("Best validation acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


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
    print(args)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5 ,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }

    data_dir = args.data_dir
    batch_size = args.batch_size
    validation_ratio = 0.2
    num_workers = 4

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                         transform=data_transforms['test'])
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
    dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
    print("Classes: ", train_dataset.classes)

    # inputs, classes = next(iter(valid_loader))
    # out = torchvision.utils.make_grid(inputs)
    # imsave(inputs, 'example_image.png')

    model = models.resnet101(pretrained=True)
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, len(train_dataset.classes))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, args.num_epochs)

    torch.save(model, os.path.join(args.save_dir, 'gan-detection-resnet101.h5'))


if __name__ == '__main__':
    main()