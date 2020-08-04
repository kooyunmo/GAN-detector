import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms


def preprocess(trainset_path, testset_path, num_workers=4, batch_size=32, validation_ratio=0.2):
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

    print("Classes: ", train_dataset.classes)
    print("==== Dataset Size ====")
    print("Training set: ", len(train_dataset))
    print("Test set: ", len(test_dataset))
    print()

    return train_loader, valid_loader, test_loader
