import torch

import numpy as np
from DnCnn import DnCNN
import os, datetime
import cv2
from skimage.measure import compare_psnr, compare_ssim
import argparse
import matplotlib.pyplot as plt
import glob


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='dir of test dataset')
    parser.add_argument('--set_names', default=['Set12','set68'], help='names of training datasets')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default='modeldata/', type=str, help='dir of the model')
    parser.add_argument('--model_name', default='model_009.pth', type=str, help='name of the model')
    # parser.add_argument('--result_dir',default='results',type=str, help = 'dir of test result data')
    # parser.add_argument('--save_result',default=0,type=int,help = 'save the result image')

    return parser.parse_args()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == "__main__":
    # create and initalization model
    modellist = glob.glob(os.path.join('modeldata',"model_*.pth"))
    args = parser_arguments()

    dncnn_test = DnCNN()
    dncnn_test = dncnn_test.to(device)
    count = 0
    #  print(dncnn_test)
    # load model in n'th epoch
 #  dncnn_test = torch.load(os.path.join(args.model_dir, args.model_name))  # 임시로 조치함.
   # dncnn_test.eval()
    # print("load model in %3dth epoch"%epoch)

    psnr_array = np.zeros((2,50))

    for k,setname in enumerate(args.set_names):
        # load imageset
        for i,modelname in enumerate(modellist):

            print("load {}th module {}".format(i,modelname))
            dncnn_test = torch.load(modelname)
            dncnn_test.eval()


            path_testset = os.path.join(args.set_dir, setname)
            filelist = os.listdir(path_testset)
            psnr_value = []
            ssim_value = []

            for file in filelist:
                if file.endswith('.png') or file.endswith('.bmp') or file.endswith('jpg'):
                    # load image
                    x = np.array(cv2.imread(os.path.join(path_testset, file), ), dtype=np.float32) / 255.0
                    #    print(x.shape)
                    #  plt.imshow(x)
                    # plt.show()
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                    #    print(x.shape)
                    # plt.imshow(x,'gray')
                    # plt.show()
                    x = np.expand_dims(x, axis=0)
                    #   print(x.shape)
                    y = x + np.random.randn(x.shape[0], x.shape[1], x.shape[2]) * args.sigma / 255.0
                    #  y = y.astype(np.float32)
                    y_ = torch.from_numpy(y).view(1, -1, y.shape[1], y.shape[2])
                    #   print(y_.shape)
                    torch.cuda.synchronize()
                    # x = torch.from_numpy(x).to(device)
                    y_ = y_.to(device, dtype=torch.float)

                    # inference ouput
                    out = dncnn_test(y_)
                    out = out.view(y.shape[1], y.shape[2])
                    out = out.cpu()
                    out = out.detach().numpy().astype(np.float32)
                    torch.cuda.synchronize()

                    x = np.squeeze(x)
                    PSNR = compare_psnr(out, x, 1)
                    SSIM = compare_ssim(out, x)

                    psnr_value.append(PSNR)
                    ssim_value.append(SSIM)
                    name, ext = os.path.splitext(file)

                    count += 1
                    """
                    if count % 15 == 0:
                        plt.figure()
                        plt.imshow(np.squeeze(y), 'gray')
                        plt.figure()
                        plt.imshow(out, 'gray')
                        plt.show()
                        plt.imshow(x, 'gray')
                        plt.show()
                    """
            psnr_avg = np.mean(psnr_value)
            ssim_avg = np.mean(ssim_value)

            psnr_array[k][i] = psnr_avg
            log('{}th module, Dataset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(i,setname, psnr_avg, ssim_avg))
            np.save('./psnr_test/psnr_testresult.npy',psnr_array)
