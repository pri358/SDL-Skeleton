import os
import numpy as np
import pandas as pd
import skimage.io as io
from torch.utils.data import Dataset
import torch


class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        inputName = self.frame.iloc[idx, 0]
        targetName = self.frame.iloc[idx, 1]

        inputImage = io.imread(inputName)
        if len(inputImage.shape) == 2:
            inputImage = inputImage[:, :, np.newaxis]
            inputImage = np.repeat(inputImage, 3, axis=-1)
            inputImage = inputImage[:, :, ::-1]
            inputImage = inputImage.astype(np.float32)
            inputImage -= np.array([104.00699 / 3, 116.66877 / 3, 122.67892 / 3])
            inputImage = inputImage.transpose((2, 0, 1))
        else:
            inputImage = inputImage[:, :, ::-1]
            inputImage = inputImage.astype(np.float32)
            inputImage -= np.array([104.00699, 116.66877, 122.67892])
            inputImage = inputImage.transpose((2, 0, 1))

        targetImage = io.imread(targetName)
        if len(targetImage.shape) == 3:
            targetImage = targetImage[:, :, 0]
        targetImage = targetImage > 0.0
        targetImage = targetImage.astype(np.float32)
        targetImage = np.expand_dims(targetImage, axis=0)

        inputImage = torch.Tensor(inputImage)
        targetImage = torch.Tensor(targetImage)
        return inputImage, targetImage


class TestDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = sorted(os.listdir(rootDir))
        # self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ', header=None)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        fname = self.frame[idx]
        inputName = os.path.join(self.rootDir, fname)

        inputImage = io.imread(inputName)[:, :, ::-1]
        inputImage = inputImage.astype(np.float32)
        inputImage -= np.array([104.00699, 116.66877, 122.67892])
        inputImage = inputImage.transpose((2, 0, 1))

        inputImage = torch.Tensor(inputImage)

        return inputImage, fname
