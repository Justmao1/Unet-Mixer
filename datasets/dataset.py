import os
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
    def __getitem__(self, index):
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
