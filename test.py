import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import mymodel
from datasets.dataset import MyValueDataSet
from utils.metrics import torchPSNR, torchSSIM

if __name__ == '__main__':
    os.makedirs("./data/result", exist_ok=True)
    resultPathTest = "./data/result"
    inputPathTest = "./data/test/DATA_noisy5"
    targetPathTest = "./data/test/DATA_clean"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load model
    myNet = mymodel(in_channels=1).to(device)
    myNet.load_state_dict(torch.load('./model_best.pth', map_location=device))
    myNet.eval()

    # Load test dataset
    datasetTest = MyValueDataSet(inputPathTest, targetPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=True)

    testpsnr = []
    testssim = []
    print('--------------------------------------------------------------')
    with torch.no_grad():
        timeStart = time.time()
        for index, (x, y) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()
            input_test, target = x.to(device), y.to(device)
            output_test = myNet(input_test)

            # Convert output to grayscale (single channel)
            output_test = output_test.mean(dim=1, keepdim=True)

            # Calculate metrics
            testpsnr.append(torchPSNR(target, output_test))
            testssim.append(torchSSIM(target, output_test))

            # Save output image
            save_image(output_test, os.path.join(resultPathTest, str(index + 1).zfill(3) + '.png'))

        # Average metrics
        out_psnr = torch.stack(testpsnr).mean().item()
        out_ssim = np.mean(testssim)
        timeEnd = time.time()

        print('---------------------------------------------------------')
        print(f"Testing Finished !!! Time: {timeEnd - timeStart:.4f} s, PSNR: {out_psnr:.2f}, SSIM: {out_ssim:.4f}")
