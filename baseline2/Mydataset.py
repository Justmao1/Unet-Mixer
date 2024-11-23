import os
import sys
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
from tqdm import tqdm


# 基础数据集类
class BaseDataset(Dataset):
    def __init__(self, inputPath, targetPath=None):
        self.inputPath = inputPath
        self.inputImages = glob(os.path.join(inputPath, '*'))  # 获取输入图像列表
        self.targetPath = targetPath
        if targetPath:
            self.targetImages = glob(os.path.join(targetPath, '*'))  # 获取目标图像列表

    def __len__(self):
        return len(self.inputImages)  # 返回输入图像的数量

    def load_image(self, path):
        return ttf.to_tensor(Image.open(path))  # 加载图像并转换为张量


# 训练数据集类
class MyTrainDataSet(BaseDataset):
    def __getitem__(self, index):
        inputImage = self.load_image(self.inputImages[index])  # 加载输入图像
        targetImage = self.load_image(self.targetImages[index])  # 加载目标图像
        return inputImage, targetImage  # 返回输入图像和目标图像


# 验证数据集类
class MyValueDataSet(BaseDataset):
    def __getitem__(self, index):
        inputImage = self.load_image(self.inputImages[index])  # 加载输入图像
        targetImage = self.load_image(self.targetImages[index])  # 加载目标图像
        return inputImage, targetImage  # 返回输入图像和目标图像


# 测试数据集类
class MyTestDataSet(BaseDataset):
    def __init__(self, inputPath):
        super().__init__(inputPath)

    def __getitem__(self, index):
        inputImage = self.load_image(self.inputImages[index])  # 加载输入图像
        return inputImage  # 返回输入图像

