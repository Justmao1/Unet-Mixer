import os
import random
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf


class BaseDataset(Dataset):
    def __init__(self, inputPath, targetPath=None):
        self.inputPath = inputPath
        self.inputImages = sorted(glob(os.path.join(inputPath, '*')))
        self.targetPath = targetPath
        if targetPath:
            self.targetImages = sorted(glob(os.path.join(targetPath, '*')))

    def __len__(self):
        return len(self.inputImages)

    def load_image(self, path):
        return ttf.to_tensor(Image.open(path))


class MyTrainDataSet(BaseDataset):
    """Training dataset with optional on-the-fly Gaussian noise generation.

    When noise_level is set, loads clean images and adds Gaussian noise
    on-the-fly to generate noisy inputs. When noise_level is None (default),
    loads pre-existing noisy/clean image pairs from disk.
    """

    def __init__(self, inputPath, targetPath=None, noise_level=None):
        # When using online noise, inputPath should point to clean images
        # and targetPath can be None (clean images are used as both input and target)
        super().__init__(inputPath, targetPath)
        self.noise_level = noise_level

    def add_gaussian_noise(self, image, sigma):
        """Add Gaussian noise to a clean image tensor."""
        noise = torch.randn_like(image) * (sigma / 255.0)
        noisy = image + noise
        return torch.clamp(noisy, 0.0, 1.0)

    def __getitem__(self, index):
        if self.noise_level is not None:
            # Online noise mode: load clean image, add noise on-the-fly
            clean = self.load_image(self.inputImages[index])
            sigma = self.noise_level if not isinstance(self.noise_level, tuple) \
                else random.uniform(self.noise_level[0], self.noise_level[1])
            noisy = self.add_gaussian_noise(clean, sigma)
            return noisy, clean
        else:
            # Pre-noised mode: load existing noisy/clean pairs
            inputImage = self.load_image(self.inputImages[index])
            targetImage = self.load_image(self.targetImages[index])
            return inputImage, targetImage


class MyValueDataSet(BaseDataset):
    def __getitem__(self, index):
        inputImage = self.load_image(self.inputImages[index])
        targetImage = self.load_image(self.targetImages[index])
        return inputImage, targetImage


class MyTestDataSet(BaseDataset):
    def __init__(self, inputPath):
        super().__init__(inputPath)

    def __getitem__(self, index):
        inputImage = self.load_image(self.inputImages[index])
        return inputImage
