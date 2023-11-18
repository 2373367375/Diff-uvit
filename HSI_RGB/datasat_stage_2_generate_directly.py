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

    def __init__(self, name1, name2, REF, data, mode, channel=128, padding = 2, nums = 500):
        super(Dataset, self).__init__()
        self.channel = channel
        self.mode = mode
        self.filename_T1 = name1
        self.filename_T2 = name2
        self.REF = REF
        self.nums = nums
        self.padding = padding
        self.filename_T2_Mat = './mat_file/2000_1000_0_layer_f/'
        self.Data_all = {}
        self.unpsample = nn.Upsample(scale_factor=3, mode='bicubic')


        if data == "Data_4":

            file = sorted(os.listdir(self.filename_T2_Mat))
            file = list(reversed(file))


            self.Data_all['T1_f_step0_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step0_0']
            self.Data_all['T1_f_step0_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step0_1']
            self.Data_all['T1_f_step0_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step0_2']
            self.Data_all['T1_f_step0_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step0_3']
            self.Data_all['T1_f_step0_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step0_4']

            self.Data_all['T1_f_step1000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step1000_0']
            self.Data_all['T1_f_step1000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step1000_1']
            self.Data_all['T1_f_step1000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step1000_2']
            self.Data_all['T1_f_step1000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step1000_3']
            self.Data_all['T1_f_step1000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step1000_4']
            self.Data_all['T1_f_step1000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step1000_6']

            self.Data_all['T1_f_step2000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step2000_0']
            self.Data_all['T1_f_step2000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step2000_1']
            self.Data_all['T1_f_step2000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step2000_2']
            self.Data_all['T1_f_step2000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step2000_3']
            self.Data_all['T1_f_step2000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step2000_4']
            self.Data_all['T1_f_step2000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_f_step2000_6']
            self.Data_all['T1_x_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T1_x_0']

            self.Data_all['T2_f_step0_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step0_0']
            self.Data_all['T2_f_step0_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step0_1']
            self.Data_all['T2_f_step0_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step0_2']
            self.Data_all['T2_f_step0_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step0_3']
            self.Data_all['T2_f_step0_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step0_4']

            self.Data_all['T2_f_step1000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step1000_0']
            self.Data_all['T2_f_step1000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step1000_1']
            self.Data_all['T2_f_step1000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step1000_2']
            self.Data_all['T2_f_step1000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step1000_3']
            self.Data_all['T2_f_step1000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step1000_4']
            self.Data_all['T2_f_step1000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step1000_6']

            self.Data_all['T2_f_step2000_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step2000_0']
            self.Data_all['T2_f_step2000_1'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step2000_1']
            self.Data_all['T2_f_step2000_2'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step2000_2']
            self.Data_all['T2_f_step2000_3'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step2000_3']
            self.Data_all['T2_f_step2000_4'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step2000_4']
            self.Data_all['T2_f_step2000_6'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_f_step2000_6']
            self.Data_all['T2_x_0'] = loadmat(os.path.join(self.filename_T2_Mat, file.pop()))['T2_x_0']

            self.Data_all['T1_x_org'] = loadmat(os.path.join(self.filename_T1))['T1'][0: 27 * 27, 0: 27 * 13, :]
            self.Data_all['T2_x_org'] = self.unpsample(torch.from_numpy(loadmat(os.path.join(self.filename_T2))['T2']).permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)[0:27*27, 0:27*13, :].numpy()

            self.image_REF = loadmat(os.path.join(self.REF))['gt']
            self.image_REF = self.image_REF[0:27*27, 0:27*13]
            # self.image_REF[0:280, 110:400] = 3

        self.h, self.w = self.Data_all['T1_f_step0_0'].shape[0], self.Data_all['T1_f_step0_0'].shape[1]
        for i in self.Data_all.keys():
            self.Data_all[i] = cv.copyMakeBorder(self.Data_all[i], self.padding, self.padding, self.padding, self.padding, cv.BORDER_REFLECT)

        all_num = self.h * self.w
        random.seed(1)
        self.whole_point = self.image_REF.reshape(1, all_num)
        if self.mode == "train":
            self.Changed_point = random.sample(list(np.where(self.whole_point[0] == 1)[0]), int(len(np.where(self.whole_point[0] == 1)[0])*0.1))
            self.NChanged_point = random.sample(list(np.where(self.whole_point[0] == 2)[0]), int(len(np.where(self.whole_point[0] == 2)[0])*0.1))
            self.random_point = self.Changed_point+self.NChanged_point
        if self.mode == "test":
            self.random_point = list(range(all_num))

        self.device = torch.device("cuda:0")

    def __len__(self):
        return len(self.random_point)

    def __getitem__(self, index):

        original_i = int((self.random_point[index] / self.w))
        original_j = (self.random_point[index] - original_i * self.w)
        new_i = original_i + self.padding
        new_j = original_j + self.padding
        self.Data_patch = {}
        for i in self.Data_all.keys():
            self.Data_patch[i+'_patch'] = self.Data_all[i][new_i - self.padding: new_i + self.padding+1,
                               new_j - self.padding: new_j + self.padding+1].transpose(2, 0, 1)

        GT = self.image_REF[original_i, original_j]

        return self.Data_patch, GT-1


# if __name__ == "__main__":
#     db = Dataset('result/{0}/coe_T1.mat'.format(8),'result/{0}/coe_T2.mat'.format(8),'data/REF.mat', 'train', channel = 8, padding = 2)
#     train_data = DataLoader(db, batch_size = 16, shuffle = False)
#     for step, (image_1,image_2, label) in enumerate(train_data):
#         print("step: %d" %(step + 1), image_1.shape, image_2.shape, label.shape)