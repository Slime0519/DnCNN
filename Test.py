import torch

import numpy as np
from DnCnn import DnCNN
import os, datetime
import cv2
from skimage.measure import compare_psnr,compare_ssim

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


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == "__main__":
    #create and initalization model
    dncnn_test = DnCNN()
    dncnn_test.to(device)
    count = 0
    for epoch in range(0,1):
        #load model in n'th epoch
        dncnn_test = torch.load(os.path.join(modeldir, 'model_050.pth'))#임시로 조치함.
        dncnn_test.eval()
        #print("load model in %3dth epoch"%epoch)

        #load imageset
        path_testset = os.path.join('./testsets',imageset_name)
        filelist =os.listdir(path_testset)
        print(filelist)
        for file in filelist:
            if file.endswith('.png') or file.endswith('.bmp') or file.endswith('jpg'):
                #load image
                x = np.array(cv2.imread(os.path.join(path_testset,file),),dtype=np.float32)/255.0
            #    print(x.shape)
              #  plt.imshow(x)
               # plt.show()
                x= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            #    print(x.shape)
               # plt.imshow(x,'gray')
                #plt.show()
                x = np.expand_dims(x,axis=0)
             #   print(x.shape)
                y = x + np.random.randn(x.shape[0],x.shape[1],x.shape[2])* sigma / 255.0
              #  y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[1], y.shape[2])
             #   print(y_.shape)
                torch.cuda.synchronize()
                #x = torch.from_numpy(x).to(device)
                y_ = y_.to(device,dtype = torch.float)

                #inference ouput
                out = dncnn_test(y_)
                out = out.view(y.shape[1],y.shape[2])
                out = out.cpu()
                out = out.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()

                x= np.squeeze(x)
                PSNR = compare_psnr(out,x,1)
                SSIM = compare_ssim(out,x)

                psnr_value.append(PSNR)
                ssim_value.append(SSIM)
                name, ext = os.path.splitext(file)

                count+=1
                if count%15 == 0:
                    plt.figure()
                    plt.imshow(np.squeeze(y),'gray')
                    plt.figure()
                    plt.imshow(out,'gray')
                    plt.show()
                    plt.imshow(x,'gray')
                    plt.show()
        psnr_avg = np.mean(psnr_value)
        ssim_avg = np.mean(ssim_value)
        log('Dataset: {0:10d} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(epoch, psnr_avg, ssim_avg))



