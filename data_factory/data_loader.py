import os.path
import random
import shutil

from glob import glob
import torchvision.datasets as dset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_dataset(data_path='dataset', dataset='pepe', batch_size=4, img_size=36):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        dataset = dsets.MNIST(root=f'{data_path}/MNIST_data/',
                              train=True,
                              transform=transform,
                              download=True)

    else:
        dataset = dset.ImageFolder(root=f'{data_path}/{dataset}',
                                   transform=transforms.Compose([
                                       transforms.Resize(img_size),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5),
                                   ]))

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

