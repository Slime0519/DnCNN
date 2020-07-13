import numpy as np
import cv2
import glob
import torch
import torch.utils.data as udata
import matplotlib.pyplot as plt

patch_size, stride = 40, 10
scales = [1,0.9,0.8,0.7]
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

    for ratio in scales:
        height_scaled, width_scaled = int(img_size[0]*ratio), int(img_size[1]*ratio)
        img_scaled = cv2.resize(img, (height_scaled,width_scaled),interpolation=cv2.INTER_CUBIC)
        for i in range(0,height_scaled-patch_size+1,stride):
            for j in range(0,width_scaled-patch_size+1,stride):
                cropped_img = img_scaled[i:i+patch_size,j:j+patch_size]
                cropped_img = np.float32(cropped_img/255.)

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
   # print(dataset[0])
   # plt.imshow(dataset[0])
   # plt.show()
    data = np.array(dataset)
   # plt.imshow(data[0])
   # plt.show()
    #print(dataset)
    data = np.expand_dims(data, axis=3)
    delete_range = len(data)-(len(data)//batch_size)*batch_size
    #print(delete_range)
    train_dataset = np.delete(data, range(delete_range), axis = 0)
    return train_dataset

class Dataset_Denoising():
    def __init__(self,xs,sigma):
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x +noise/500
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)
