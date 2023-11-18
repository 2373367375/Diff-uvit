import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
import cv2 as cv
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
import random
from sklearn.decomposition import PCA
from copy import deepcopy
from PIL import Image


class Dataset(nn.Module):

    def __init__(self, name1, name2, REF, data, mode, channel=128, padding = 13, nums = 500):
        super(Dataset, self).__init__()
        self.channel = channel
        self.mode = mode
        self.filename_T1 = name1
        self.filename_T2 = name2
        self.REF = REF
        self.nums = nums
        self.padding = padding

        if data == "Data_2":
            self.image_T1 = np.expand_dims(loadmat(os.path.join(self.filename_T1))['Pan_Norm'][0:324,0:270,0],2)
            # self.image_T1 = self.image_T1.reshape(row * col, -1).transpose(1,0)
            self.image_T2 = np.expand_dims(loadmat(os.path.join(self.filename_T2))['SAR_Norm'][0:324,0:270,0],2)
            # self.image_T2 = self.image_T2.reshape(row * col, -1).transpose(1, 0)
            self.image_REF = loadmat(os.path.join(self.REF))['GT'][0:324,0:270]

        self.padding_image_T1 = cv.copyMakeBorder(self.image_T1, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)
        self.padding_image_T2 = cv.copyMakeBorder(self.image_T2, self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)
        self.h, self.w = self.image_T1.shape[0], self.image_T1.shape[1]
        all_num = self.h * self.w
        random.seed(1)
        self.whole_point = self.image_REF.reshape(1, all_num)
        if self.mode == "train":
            # self.Changed_point = random.sample(list(np.where(self.whole_point[0] == 255)[0]), 500)
            self.NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), int(len(list(np.where(self.whole_point[0] == 0)[0]))*0.1))
            self.random_point = self.NChanged_point
        if self.mode == "test":
            self.random_point = list(range(all_num))
            # self.NChanged_point = random.sample(list(np.where(self.whole_point[0] == 0)[0]), 1000)
            # self.random_point = self.NChanged_point

    def __len__(self):
        return len(self.random_point)

    def __getitem__(self, index):

        original_i = int((self.random_point[index] / self.w))
        original_j = (self.random_point[index] - original_i * self.w)
        new_i = original_i + self.padding
        new_j = original_j + self.padding

        RGB = np.expand_dims(self.padding_image_T1[new_i - self.padding: new_i + self.padding+1,
        new_j - self.padding: new_j + self.padding+1],2).transpose(2, 0, 1).reshape(self.channel,
                                                                                      self.padding * 2+1,
                                                                                      self.padding * 2+1)

        SAR = np.expand_dims(self.padding_image_T2[new_i - self.padding: new_i + self.padding+1,
        new_j - self.padding: new_j + self.padding+1],2).transpose(2, 0, 1).reshape(self.channel,
                                                                                      self.padding * 2+1,
                                                                                      self.padding * 2+1)

        GT = self.image_REF[original_i, original_j]

        return RGB, SAR, GT
# if __name__ == "__main__":
#     db = Dataset('result/{0}/coe_T1.mat'.format(8),'result/{0}/coe_T2.mat'.format(8),'data/REF.mat', 'train', channel = 8, padding = 2)
#     train_data = DataLoader(db, batch_size = 16, shuffle = False)
#     for step, (image_1,image_2, label) in enumerate(train_data):
#         print("step: %d" %(step + 1), image_1.shape, image_2.shape, label.shape)