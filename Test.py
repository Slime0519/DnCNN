import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DnCnn import DnCNN
import os
import cv2
from skimage.measure import compare_psnr, compare_ssim

import matplotlib.pyplot as plt

num_epoch = 50
sigma = 25
psnr_value = []
ssim_value = []

modeldir = './modeldata/'
imageset_name = 'BSD68'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if __name__ == "__main__":
    #create and initalization model
    dncnn_test = DnCNN()
    dncnn_test.to(device)

    for epoch in range(0,num_epoch):
        #load model in n'th epoch
        dncnn_test = torch.load(os.path.join(modeldir, 'model_%3d.pth' % epoch))
        dncnn_test.eval()
        print("load model in %3dth epoch"%epoch)

        #load imageset
        path_testset = os.path.join('./testsets',imageset_name)
        filelist =os.listdir(path_testset)

        for file in filelist:
            if file.endswith('.png') or file.endswith('.bmp') or file.endswith('jpg'):
                #load image
                x = np.array(cv2.imread(os.path.join(path_testset,file),),dtype='float32')/255.0
                y = x + np.random.randn(0, sigma / 255.0, x.shape())
                x = torch.from_numpy(y).view(1,-1,y.shape[0],y.shape[1])

                torch.cuda.synchronize()
                #x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)

                #inference ouput
                out = dncnn_test(y)
                out = out.view(y.shape[0],y.shape[1])
                out = out.cpu()
                out = out.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()

                PSNR = compare_psnr(out,x)
                SSIM = compare_ssim(out,x)

                psnr_value.append(PSNR)
                ssim_value.append(SSIM)
                name, ext = os.path.splitext(file)
                plt.figure()
                plt.imshow(np.hstack(y,out))
                save
        psnr_avg = np.mean(psnr_value)
        ssim_avg = np.mean(ssim_value)





