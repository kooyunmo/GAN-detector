import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms


def preprocess(model_name, data_dir, num_workers=4, batch_size=32, validation_ratio=0.1):
    transform_size = 224 if 'resnet' in model_name else 299

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(transform_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test'])

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
    print("==== Dataset Size ====")
    print("Training set: ", len(train_dataset))
    print("Test set: ", len(test_dataset))
    print()

    return dataloaders
