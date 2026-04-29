import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import mymodel
from datasets.dataset import MyTrainDataSet, MyValueDataSet
from utils.metrics import torchPSNR
from utils.lr_scheduler import CyclicLR

if __name__ == '__main__':
    EPOCH = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    loss_list = []
    best_psnr = 0
    best_epoch = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    inputPathTrain = "./data/train/DATA_noisy5"
    targetPathTrain = "./data/train/DATA_clean"
    inputPathVal = "./data/val/DATA_noisy5"
    targetPathVal = "./data/val/DATA_clean"
    inputPathTest = "./data/test/DATA_noisy5"
    targetPathTest = "./data/test/DATA_clean"
    resultPathTest = "./data/result"

    # Online noise config: set to a sigma value (e.g., 5) to generate noise on-the-fly
    # Set to None to use pre-existing noisy/clean image pairs
    USE_ONLINE_NOISE = False
    NOISE_LEVEL = 5  # Only used when USE_ONLINE_NOISE is True

    os.makedirs(resultPathTest, exist_ok=True)

    # Model
    myNet = mymodel(in_channels=1).to(device)
    criterion = nn.MSELoss().to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, myNet.parameters()), lr=LEARNING_RATE)
    scheduler = CyclicLR(optimizer, base_lr=LEARNING_RATE, max_lr=4 * LEARNING_RATE,
                         step_size=2000, mode='exp_range', gamma=0.99994)

    # Data
    noise_sigma = NOISE_LEVEL if USE_ONLINE_NOISE else None
    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain, noise_level=noise_sigma)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=False, num_workers=0, pin_memory=True)

    datasetValue = MyValueDataSet(inputPathVal, targetPathVal)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=16, shuffle=False,
                             drop_last=False, num_workers=0, pin_memory=True)

    datasetTest = MyValueDataSet(inputPathTest, targetPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=True)

    # Resume from checkpoint if exists
    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists('./model_best.pth'):
        myNet.load_state_dict(torch.load('./model_best.pth', map_location=device), strict=False)
        print("Loaded pretrained model from model_best.pth")

    # Training
    for epoch in range(EPOCH):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()

        for index, (x, y) in enumerate(iters, 0):
            myNet.zero_grad()
            optimizer.zero_grad()

            input_train, target = x.to(device), y.to(device)
            output_train = myNet(input_train)
            loss = criterion(output_train, target)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch + 1, EPOCH, loss.item()))

        # Validation
        myNet.eval()
        psnr_val_rgb = []
        for index, (x, y) in enumerate(valueLoader, 0):
            input_, target_value = x.to(device), y.to(device)
            with torch.no_grad():
                output_value = myNet(input_)
            for output_v, target_v in zip(output_value, target_value):
                psnr_val_rgb.append(torchPSNR(output_v, target_v))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(myNet.state_dict(), './model_best.pth')

        loss_list.append(epochLoss)
        torch.save(myNet.state_dict(), 'model.pth')
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch + 1, timeEnd - timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')

    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

    # Testing with best model
    print('--------------------------------------------------------------')
    myNet.load_state_dict(torch.load('./model_best.pth', map_location=device))
    myNet.eval()

    testpsnr = []
    with torch.no_grad():
        timeStart = time.time()
        for index, (x, y) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            input_test, target = x.to(device), y.to(device)
            output_test = myNet(input_test)
            testpsnr.append(torchPSNR(target, output_test))
            save_image(output_test, os.path.join(resultPathTest, str(index + 1).zfill(3) + '.tif'))
        out_psnr = torch.stack(testpsnr).mean().item()
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s, Best PSNR : {:.2f}".format(timeEnd - timeStart, out_psnr))

    # Plot loss curve
    plt.figure(1)
    x = range(0, EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.plot(x, loss_list, 'r-')
    plt.savefig('loss_curve.png')
    plt.show()
