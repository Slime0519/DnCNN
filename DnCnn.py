import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
#def getnoise(noise_level = 25):


patch_size, stride = 40, 10
aug_number = 1
batch_size = 128

def data_aug(img):

    num = np.random.randint(0,8)
    if num == 0:
        return img
    elif num ==1:
        return np.flipud(img)
    elif num ==2:
        return np.rot90(img)
    elif num ==3:
        return np.flipud(np.rot90(img))
    elif num ==4:
        return np.rot90(img,k=2)
    elif num ==5:
        return np.flipud(np.rot90(img,k=2))
    elif num ==6:
        return np.rot90(img,k=3)
    elif num ==7:
        return np.flipud(np.rot90(img,k=3))


def makepatches(file_name):
    img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE) #grayscale
    patches = []
    img_size = img.shape

    for i in range(0,img_size[0]-patch_size+1,stride):
        for j in range(0,img_size[1]-patch_size+1,stride):
            cropped_img = img[i:i+patch_size,j:j+patch_size]
            for k in range(aug_number):
                cropped_img = data_aug(cropped_img)
                patches.append(cropped_img)

    return patches

def trainimage_generator(dirpath):
    filenames = glob.glob(dirpath+"/*.png")

    dataset = []

    for filename in filenames:
        patchset = makepatches(filename)
        for patch in patchset:
            dataset.append(patch)

    data = np.array(dataset,dtype = 'uint8')
    data = np.expand_dims(data, axis=3)
    delete_range = len(data)-(len(data)//batch_size)*batch_size
    print(delete_range)
    train_dataset = np.delete(data, range(delete_range), axis = 0)
    return train_dataset

class Dataset_Denoising():
    def __init__(self,xs,sigma):
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x +noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)



class DnCNN(nn.Module):
    def __init__(self,iscolor = 'False'):
        super(DnCNN, self).__init__()
        #def type 1(first) layer
        if(iscolor == 'False'):
            color_channel = 1
        else:
            color_channel = 3
        self.type1_conv = nn.Conv2d(in_channels=color_channel,out_channels=64,kernel_size=3,padding='same', padding_mode='zeros',bias='True')

        #def type2 layer
        self.type2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', padding_mode= 'zeros',bias='True')
        self.type2_bn = nn.BatchNorm2d(64,eps = 0.0001, momentum = 0.95)

        #def type3 layer
        self.type3_conv = nn.Conv2d(in_channels=64,out_channels=color_channel, kernel_size=3 ,padding='same', padding_mode='zeros',bias='True')

    def make_model(self,depth):
        modules = []
        modules.append(self.type1_conv)
        modules.append(F.relu)
        for i in range(depth-2):
            modules.append(self.type2_conv)
            modules.append(self.type2_bn)
            modules.append(F.relu())
        modules.append(self.type3_conv)

        model = nn.Sequential(*modules)
        return model

    def forward(self, x, depth):
        model = self.make_model(depth)
        out = model(x)
        return out


if __name__ == '__main__':
    #create training dataset
    patches = makepatches('./data/Train400')
    train_dataset = Dataset_Denoising(patches, sigma=25)

    #create model
    Dncnn = DnCNN(iscolor='false', )
    Dncnn.train()
    optimizer = optim.Adam(Dncnn.parameters(), lr = )


    criterion = nn.MSELoss()
    loss = criterion()
    for
    output = Dncnn(input)

    trainset = trainimage_generator('./data/Train400')
    print(trainset.shape)



#plt.imshow(np.reshape(trainset[400],(40,40)))
#plt.show()
#

