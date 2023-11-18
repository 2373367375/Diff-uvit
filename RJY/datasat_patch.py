import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
import cv2 as cv
from torch.utils.data import DataLoader
import os

def Split_Patches(image):

    # 获取输入图像的宽度和高度
    width, height, band = image.shape
    patch_size = 27
    # 检查输入图像尺寸是否合适
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError("输入图像尺寸不是27的倍数，无法均匀切分。")

    # 计算需要切分的行数和列数
    num_rows = height // patch_size
    num_cols = width // patch_size
    ALL_Patch = []
    # 循环切分图像并存储patch
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前patch的位置
            left = col * patch_size
            upper = row * patch_size
            right = left + patch_size
            lower = upper + patch_size

            patch = image[left: right, upper: lower, :]
            ALL_Patch.append(patch)
    return ALL_Patch

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
            self.image_T1 = loadmat(os.path.join(self.filename_T1))['LRHSt1_norm'][0:270, 0:216, :]
            # self.image_T1 = self.image_T1.reshape(row * col, -1).transpose(1,0)
            self.image_T2 = loadmat(os.path.join(self.filename_T2))['rgb'][0:270, 0:216, :]
            # self.image_T2 = self.image_T2.reshape(row * col, -1).transpose(1, 0)
            self.image_REF = loadmat(os.path.join(self.REF))['Binary'][0:270, 0:216]

        self.padding_image_T1 = self.image_T1
        self.padding_image_T2 = self.image_T2
        self.h, self.w = self.image_T1.shape[0], self.image_T1.shape[1]
        all_num = self.h * self.w
        self.Patch_T1 = Split_Patches(self.padding_image_T1)
        self.Patch_T2 = Split_Patches(self.padding_image_T2)


    def __len__(self):
        return len(self.Patch_T2)

    def __getitem__(self, index):

        return self.Patch_T1[index].transpose(2,0,1), self.Patch_T2[index].transpose(2,0,1), 0
# if __name__ == "__main__":
#     db = Dataset('result/{0}/coe_T1.mat'.format(8),'result/{0}/coe_T2.mat'.format(8),'data/REF.mat', 'train', channel = 8, padding = 2)
#     train_data = DataLoader(db, batch_size = 16, shuffle = False)
#     for step, (image_1,image_2, label) in enumerate(train_data):
#         print("step: %d" %(step + 1), image_1.shape, image_2.shape, label.shape)