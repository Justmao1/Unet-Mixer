import sys
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import mymodel
from Mydataset import *
from utils import torchPSNR, torchSSIM
import time

if __name__ == '__main__':
    if not os.path.exists("./data/result"):
        os.makedirs("./data/result")
    resultPathTest = "./data/result"  # Path to save test results
    inputPathTest = "./data/test/DATA_noisy5"  # Test input image path
    targetPathTest = "./data/test/DATA_clean"  # Test target image path
    device = 'cuda:0'  # if torch.cuda.is_available() else 'cpu'

    # Instantiate and load the pre-trained model
    myNet = mymodel(in_channels=1)  # Adjust `in_channels` based on your model
    myNet = myNet.to(device)
    myNet.load_state_dict(torch.load('./model_best.pth', map_location=device))
    myNet.eval()  # Set model to evaluation mode

    # Load test dataset
    datasetTest = MyValueDataSet(inputPathTest, targetPathTest)  # Replace with your test dataset class
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            pin_memory=True)

    testpsnr = []
    testssim = []
    print('--------------------------------------------------------------')
    with torch.no_grad():  # No need for gradients during testing
        timeStart = time.time()  # Start time for testing
        for index, (x, y) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # Clear GPU memory
            input_test, target = x.to(device), y.to(device)  # Move data to GPU or CPU
            output_test = myNet(input_test)  # Forward pass

            # Convert output to grayscale (single channel)
            output_test = output_test.mean(dim=1, keepdim=True)

            # Calculate metrics
            testpsnr.append(torchPSNR(target, output_test))
            testssim.append(torchSSIM(target, output_test))  # Assuming torchSSIM supports single channel

            # Save the output image in grayscale
            save_image(output_test, resultPathTest + '/' + str(index + 1).zfill(3) + '.png')

        # Calculate average PSNR and SSIM
        out_psnr = torch.stack(testpsnr).mean().item()
        out_ssim = np.mean(testssim)
        timeEnd = time.time()  # End time for testing

        print('---------------------------------------------------------')
        print(
            f"Testing Process Finished !!! Time: {timeEnd - timeStart:.4f} s, Best PSNR : {out_psnr:.2f}, Best SSIM : {out_ssim:.4f}")
