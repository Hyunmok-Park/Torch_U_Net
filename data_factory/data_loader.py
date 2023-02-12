import os.path
import random
import shutil

from glob import glob
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Dataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        self.transform = transform

        self.labels = [file for file in sorted(glob(f"{data_path}/{mode}/label_*.npy"))]
        self.inputs = [file for file in sorted(glob(f"{data_path}/{mode}/input_*.npy"))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = np.load(self.labels[index])
        inputs = np.load(self.inputs[index])

        # 정규화
        labels = labels/255.0
        inputs = inputs/255.0

        # 이미지와 레이블의 차원 = 2일 경우(채널이 없을 경우, 흑백 이미지), 새로운 채널(축) 생성
        if labels.ndim == 2:
            labels = labels[np.newaxis, :, :]
        if inputs.ndim == 2:
            inputs = inputs[np.newaxis, :, :]

        return inputs, labels


def load_dataset(data_path='dataset', mode='train', batch_size=4, img_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = Dataset(data_path, mode, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

