import torch, torchvision
from torch.nn import init
from datasat_2 import *
from tqdm import tqdm

from datasat_stage_2_generate_directly import Dataset
from uvit import *


class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=31, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.img_size = img_size
        self.LR_size = LR_size

        # model_RGB_to_SAR = UViT(27, 1, 6)
        # model_SAR_to_RGB = UViT(27, 1, 6)

        # self.sr3 = Diffusion(model_RGB_to_SAR, model_SAR_to_RGB, device, img_size, LR_size, out_channel)
        self.Net = CD_Net().to(device)
        # Apply weight initialization & set loss & set noise schedule
        # self.sr3.apply(self.weights_init_orthogonal)
        # self.sr3.set_loss(loss_type)
        # self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            # self.sr3 = nn.DataParallel(self.sr3)
            self.Net = nn.DataParallel(self.Net)

        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=lr)
        self.criteron = nn.CrossEntropyLoss()

        # params = sum(p.numel() for p in self.sr3.parameters())
        # print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)

            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):

        train = False

        for i in range(epoch):
            self.Net.train()
            loss_all = 0
            count = 0
            if train:
                # for step, [SAR_org, SAR, RGB_org, RGB, GT] in enumerate(tqdm(self.dataloader)):
                for step, [Data_patch, GT] in enumerate(tqdm(self.dataloader)):

                    RGB_org = RGB_org.to(self.device).type(torch.float32)
                    SAR_org = SAR_org.to(self.device).type(torch.float32)
                    # RGB = RGB.to(self.device).type(torch.float32)
                    # SAR = SAR.to(self.device).type(torch.float32)
                    GT = GT.to(self.device).type(torch.float32)

                    self.optimizer.zero_grad()
                    out = self.Net(SAR_org, SAR_org, RGB_org, RGB_org)
                    loss = self.criteron(out, GT.long())
                    loss.backward()
                    self.optimizer.step()
                    loss_all = loss_all+loss.item()
                    count = count + 1
                    # print('loss_Classification: %f' % (loss.item()))
                if i % 10 == 0:
                    self.CD_save(os.getcwd(), i)
                print('loss_Classification: %f' % (loss_all/count))
            else:
                self.sr3.eval()
                checkpoint = torch.load(self.load_path)
                print("min_loss:", checkpoint['best_val'])
                path = '/media/xd132/USER/HJ/change_detection/Bar/'
                if not os.path.exists('result'):
                    os.makedirs('result')
                outimage = np.zeros((1, 540*810))
                count = 0
                for step, [SAR_org, SAR, RGB_org, RGB, GT] in enumerate(tqdm(self.testloader)):
                    # [SAR_org, SAR, RGB_org, RGB, GT] = test_data
                    RGB_org = RGB_org.to(self.device).type(torch.float32)
                    SAR_org = SAR_org.to(self.device).type(torch.float32)
                    RGB = RGB.to(self.device).type(torch.float32)
                    SAR = SAR.to(self.device).type(torch.float32)
                    batch = SAR.shape[0]
                    GT = GT.to(self.device).type(torch.float32)
                    b, c, h, w = RGB.shape

                    RGB = RGB
                    SAR = SAR

                    out = self.Net(SAR_org, SAR, RGB_org, RGB)
                    a = out.argmax(dim=1)[0]
                    outimage[0, count:(count + batch)] = a.detach().cpu().numpy()
                    print('loss_Classification: %f' % (loss.item()))


    def test(self, RGB, SAR):

        RGB = RGB
        SAR = SAR

        self.Net.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(RGB, SAR)
            else:
                result_SR = self.sr3.super_resolution(RGB, SAR)
        self.sr3.train()
        return result_SR

    def save(self, save_path, i):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path+'model_epoch-{}.pt'.format(i))

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")

    def CD_save(self, save_path, i):
        network = self.Net
        if isinstance(self.Net, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path + 'CD_model_epoch-{}.pt'.format(i))


if __name__ == "__main__":
    batch_size = 32
    LR_size = 32
    img_size = 128
    root = '../data/data_1_RGB_SAR/'

    # train_data = Dataset(root+"RGB_Norm.mat", root+"SAR_Norm.mat", root+"GT.mat",data='Data_1', mode='train', channel=3)
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # test_data = Dataset(root+"RGB_Norm.mat", root+"SAR_Norm.mat", root+"GT.mat",data='Data_1', mode='test', channel=3)
    # test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    train_data = Dataset(root+"Q1.mat", root+"Q2.mat", root+"REF.mat",data='Data_1', mode='train', channel=3)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = Dataset(root+"Q1.mat", root+"Q2.mat", root+"REF.mat",data='Data_1', mode='test', channel=3)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1',
              dataloader=train_dataloader, testloader=test_dataloader, schedule_opt=schedule_opt,
              save_path='../CD_model/Data_1/',
              load_path='../CD_model/Data_1/CD_model_epoch-280.pt', load=True,
              inner_channel=64,
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-4, distributed=False)
    sr3.train(epoch=10000, verbose=1)



