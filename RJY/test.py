import torch
import torch.nn as nn
from model import *
import torch.optim as optim
from datasat_2 import Dataset
import numpy as np
import os
import cv2
import random
from scipy.io import savemat
import warnings
import time
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

def Patches_to_image(input):
    # 获取输入图像的宽度和高度
    patch_size = 27
    # image = torch.zeros(512, 832, 3)
    image = np.zeros((270, 216))
    width, height = image.shape
    i = 0
    # 检查输入图像尺寸是否合适
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError("输入图像尺寸不是64的倍数，无法均匀切分。")

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
            image[left: right, upper: lower] = input[i]
            i += 1

    return image

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
device = torch.device("cuda:0")
warnings.filterwarnings("ignore")
root = '/media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/HJ/change_detection/SUnet/SUNet-change_detection-main/HSI multi model CD/USA/'

def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def focal_loss(logit, target, gamma=2, alpha=0.25):
    n, c, h, w = logit.size()
    criterion = nn.CrossEntropyLoss()
    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    return loss


def train_epoch(epoch, model, optimizer, criteron, l1, train_loader, show_interview=3):
    model.train()
    loss_all, count = 0, 0
    for step, [T1, T2, GT] in enumerate(tqdm(train_loader)):
        T1 = T1.to(device).type(torch.float32)
        T2 = T2.to(device).type(torch.float32)
        GT = GT.to(device).type(torch.float32)

        optimizer[0].zero_grad()
        [total_AE_loss, disc_code_loss, disc_loss] = model(T1, T2, GT)
        total_AE_loss.backward()
        optimizer[0].step()

        optimizer[1].zero_grad()
        [total_AE_loss, disc_code_loss, disc_loss] = model(T1, T2, GT)
        disc_code_loss.backward()
        optimizer[1].step()

        optimizer[2].zero_grad()
        [total_AE_loss, disc_code_loss, disc_loss] = model(T1, T2, GT)
        disc_loss.backward()
        optimizer[2].step()

        count = count + 1

    return float(total_AE_loss / count), float(disc_code_loss / count), float(disc_loss / count)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(max_epoch, batchsz, lr):
    set_seed(0)

    train_data = Dataset(root+"T1/T1HSI.mat", root+"T2/T2RGB.mat", root+'label/REF', data='Data_2', mode='train')
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    model = AceNet().to(device)

    load = False
    if load ==True:
        checkpoint = torch.load('../CD_model/Data_2/best_100.mdl')
        print("min_loss:", checkpoint['best_val'])
        model.load_state_dict(checkpoint['state_dict'])
        print('模型加载成功')
    else:
        print('未加载模型')

    W_REG = 0.001
    encoders_params = list(model.Rx.parameters()) + list(model.Py.parameters())
    decoders_params = list(model.Sz.parameters()) + list(model.Qz.parameters())

    optimizer_ae = torch.optim.Adam(encoders_params + decoders_params, lr=lr,
                                    weight_decay=W_REG)
    optimizer_e = torch.optim.Adam(encoders_params, lr=lr, weight_decay=W_REG)
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr=lr,
                                      weight_decay=W_REG)

    optimizer = [optimizer_ae, optimizer_e, optimizer_disc]
    criteron = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    best_loss = 10
    for epoch in range(max_epoch):
        epoch = epoch
        total_AE_loss, disc_code_loss, disc_loss = train_epoch(epoch, model, optimizer, criteron, l1, train_dataloader)

        if epoch % 5 == 0:
            state = dict(epoch=epoch + 1, state_dict=model.state_dict(), best_val=total_AE_loss)
            torch.save(state, "./best_{:d}.mdl".format(epoch))

        print("epoch: %d  total_AE_loss = %.7f disc_code_loss = %.7f disc_loss = %.7f" % (epoch + 1, total_AE_loss, disc_code_loss, disc_loss))


def test(model):
    model.eval()
    checkpoint = torch.load('./best_100.mdl')
    print("min_loss:", checkpoint['best_val'])
    # model.load_state_dict(checkpoint['state_dict'])
    test_data = Dataset(root+"T1/T1HSI.mat", root+"T2/T2RGB.mat", root+'label/REF', data='Data_2', mode='test',
                        channel=1)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    if not os.path.exists('result'):
        os.makedirs('result')
    GT = loadmat('/media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/HJ/change_detection/SUnet/SUNet-change_detection-main/HSI multi model CD/USA/label/REF.mat')['Binary']
    H, W = GT.shape
    outimage = np.zeros((1, H * W))
    start = time.time()
    count = 0
    patches = []
    with torch.no_grad():
        for step, [T1, T2, GT] in enumerate(tqdm(test_dataloader)):
            T1 = T1.to(device).type(torch.float32)
            T2 = T2.to(device).type(torch.float32)
            GT = GT.to(device).type(torch.float32)
            batch = GT.shape[0]

            logits= model(T1, T2, GT)
            # label = label.type(torch.float32).to(device)
            # input = torch.cat([image_1, image_2], dim=1)
            outimage[0, count:(count + batch)] = logits.argmax(dim=1).detach().cpu().numpy()
            count += batch

        outimage = outimage.reshape(H, W)
        savemat('USA.mat', {'data': outimage})
        print("save success!!!!")
    end = time.time()
    print("running time:", end - start)


if __name__ == "__main__":
    # train(50, 64, 0.00001)
    model = AceNet().to(device)
    test(model)
