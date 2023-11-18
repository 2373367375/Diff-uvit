import torch
import torch.nn as nn
from uvit import *
import torch.optim as optim
from datasat_stage_2_generate_directly import Dataset
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

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
device = torch.device("cuda:1")
warnings.filterwarnings("ignore")
root = '../data/data_2_SAR_PAN/'

def focal_loss(logit, target, gamma=2, alpha=0.25):

    criterion = nn.CrossEntropyLoss()
    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    return loss


def mse_loss_weighted(x, x_hat, weights):
    L2_Norm = torch.linalg.norm(x_hat - x, dim=1) ** 2
    weighted_L2_norm: torch.Tensor = L2_Norm * weights
    loss = weighted_L2_norm.mean()
    return loss

def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(epoch, model, optimizer, criteron, l1, train_loader, show_interview=3):
    model.train()
    loss_all, count = 0, 0
    for step, [Data_patch, GT] in enumerate(tqdm(train_loader)):
        for i in Data_patch.keys():
            Data_patch[i] = Data_patch[i].to(device).type(torch.float32)
        GT = GT.to(device).type(torch.float32)

        optimizer.zero_grad()
        out = model(Data_patch)
        # loss = criteron(out, GT.long())
        loss = focal_loss(out, GT.long())
        loss.backward()
        optimizer.step()
        loss_all = loss_all + loss.item()
        count = count + 1

    return float(loss_all / count)


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

    train_data = Dataset(root+"Pan_Norm.mat", root+"SAR_Norm.mat", root+"GT.mat", data='Data_2', mode='train', channel=1)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = CD_Net().to(device)

    load = False
    if load ==True:
        checkpoint = torch.load('../CD_model/Data_2/best_100.mdl')
        print("min_loss:", checkpoint['best_val'])
        model.load_state_dict(checkpoint['state_dict'])
        print('模型加载成功')
    else:
        print('未加载模型')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteron = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    best_loss = 10
    for epoch in range(max_epoch):
        epoch = epoch
        train_loss = train_epoch(epoch, model, optimizer, criteron, l1, train_dataloader)
        if epoch % 25 == 0:
            state = dict(epoch=epoch + 1, state_dict=model.state_dict(), best_val=train_loss)
            torch.save(state, "../CD_model/Data_2/best_{:d}.mdl".format(epoch))

        if train_loss <= best_loss:
            state = dict(epoch=epoch + 1, state_dict=model.state_dict(), best_val=train_loss)
            torch.save(state, '../CD_model/Data_2/best.mdl')
            best_loss = train_loss
        print("epoch: %d  best_loss = %.7f train_loss = %.7f" % (epoch + 1, best_loss, train_loss))


def test(model):
    model.eval()
    checkpoint = torch.load('../CD_model/Data_2/best.mdl')
    print("min_loss:", checkpoint['best_val'])
    model.load_state_dict(checkpoint['state_dict'])
    test_data = Dataset(root+"Pan_Norm.mat", root+"SAR_Norm.mat", root+"GT.mat", data='Data_2', mode='test',
                        channel=1)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    if not os.path.exists('result'):
        os.makedirs('result')
    GT = loadmat('../data/data_2_SAR_PAN/GT.mat')['GT'][0:324, 0:270]
    H,W = GT.shape
    outimage = np.zeros((1, H * W))
    start = time.time()
    count = 0
    with torch.no_grad():
        for step, [Data_patch, GT] in enumerate(tqdm(test_dataloader)):
            for i in Data_patch.keys():
                Data_patch[i] = Data_patch[i].to(device).type(torch.float32)
            GT = GT.to(device).type(torch.float32)
            batch = GT.shape[0]
            # label = label.type(torch.float32).to(device)
            # input = torch.cat([image_1, image_2], dim=1)
            logits = model(Data_patch)
            outimage[0, count:(count + batch)] = logits.argmax(dim=1).detach().cpu().numpy()
            count += batch
        outimage = outimage.reshape(H, W)
        filename = "../result/Data_2_result/result.mat"
        savemat(filename, {"output": outimage})
        print("save success!!!!")
    end = time.time()
    print("running time:", end - start)


if __name__ == "__main__":
    train(201, 64, 0.0001)
    model = CD_Net().to(device)
    test(model)
