import DnCnn
from torch.utils.data import DataLoader
from Model_DnCNN import DnCNN
import Dataset_module
import torch
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
trainset = Dataset_module.trainimage_generator('./data/Train400')
# print(trainset.shape)
trainset = trainset.astype('float32') / 255.0
#print(trainset.shape)
trainset = torch.from_numpy(trainset.transpose(0, 3, 1, 2))
#plt.imshow(trainset[0].squeeze())
#plt.show()
train_dataset = Dataset_module.Dataset_Denoising(trainset, sigma=25)
DLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, )

for i,batch in enumerate(DLoader):
    print(len(batch[0][0]))
    print(i)
    if((i*128)%1000 == 0):
        tempimage = np.array(batch[1][i][0])
        tempimage1 = np.array(batch[0][i][0])

        plt.imshow(tempimage)
        plt.show()
        plt.imshow(tempimage1)
        plt.show()
