import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import cv2
import numpy as np
import glob
import os
import re
import argparse
from Model_DnCNN import DnCNN
import Dataset_module
import matplotlib.pyplot as plt
#def getnoise(noise_level = 25):

parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type = str, help = 'choose a type of model')
parser.add_argument('--batch_size', default=128, type = int, help = 'batch size')
parser.add_argument('--train_data', default='data/Train400', type = str, help = 'path of train data')
parser.add_argument('--sigma', default=25, type = int, help = 'noise level')
parser.add_argument('--epoch', default=50, type = int, help = 'number of train epochs')
parser.add_argument('--lr', default = 1e-3, type = float, help = 'initial learning rate for Adam')
args = parser.parse_args()



batch_size = args.batch_size
sigma = args.sigma
num_epoch = args.epoch


modelpath = './modeldata'

if not os.path.exists(modelpath):
    os.mkdir(modelpath)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def findLastepoch(dirpath):
    filelist = glob.glob(os.path.join(dirpath,"model_*.pth"))
    if filelist:
        epochlist = []
        for file_ in filelist:
            pre_epoch = re.findall('\d+',file_)
            epochlist.append(int(pre_epoch[0]))
    else:
        return 0

    return max(epochlist)

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return F.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

if __name__ == '__main__':
    #create training dataset


    #create model
    Dncnn = DnCNN()
    Dncnn.to(device)

    start_epoch = findLastepoch(modelpath)
    if start_epoch>0:
        print("start from epoch %03d" % start_epoch)
        Dncnn = torch.load(os.path.join(modelpath,'model_%03d.pth' % start_epoch))

    #Dncnn = torch.load(os.path.join(modelpath, 'model.pth'))


    optimizer = optim.Adam(Dncnn.parameters(), lr = args.lr)
    #criterion = sum_squared_error()
    #loss = criterion()
    criterion = nn.MSELoss(size_average = False)
    criterion.cuda()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma = 0.1)

    trainset = Dataset_module.trainimage_generator('./data/Train400')
    # print(trainset.shape)
    trainset = trainset.astype('float32') / 255.0
    trainset = torch.from_numpy(trainset.transpose(0, 3, 1, 2))
    train_dataset = Dataset_module.Dataset_Denoising(trainset, sigma=25)
    DLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, )

    for epoch in range(start_epoch, num_epoch):
        scheduler.step(epoch)

        epochloss = 0

        for batch_num, batch in enumerate(DLoader):
            batch_x = (batch[1]).to(device)
            batch_y = (batch[0]).to(device)
            optimizer.zero_grad()
            loss = criterion(Dncnn(batch_y), batch_x) #/(batch_y.shape[0]*batch_y.shape[1])
            epochloss += loss.item()
            loss.backward()
            optimizer.step()
          #  print(batch_num)
            if batch_num % 10 ==0:
                print('%4d %4d/%4d loss = %2.4f' %(epoch+1, batch_num, trainset.size(0)//batch_size, loss.item()/batch_size))

        torch.save(Dncnn,os.path.join(modelpath,'model_%03d.pth'%(epoch+1)))







#plt.imshow(np.reshape(trainset[400],(40,40)))
#plt.show()
#

