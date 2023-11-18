import torch, torchvision
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
# from tqdm.notebook import tqdm
from functools import partial
import matplotlib.pyplot as plt
import math, os, copy
from datasat_2 import *
from tqdm import tqdm
from prettytable import PrettyTable
import scipy.io as sio
import imgvision as iv
from uvit import *
import torch
import torch.nn as nn
import warnings
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:1', choices=['cuda:0', 'cuda:1'])
parser.add_argument('--data', default='Data_4')
parser.add_argument('--data_path', default='../data/data_4_HSI_RGB/')
parser.add_argument('--save_path', default='../TS_model/Data_4_HSI_RGB/feature_5/')
parser.add_argument('--load_path', default='../TS_model/Data_4_HSI_RGB/feature_5/model_epoch-139.pt')


class Diffusion(nn.Module):
    def __init__(self, model_HSI_to_RGB, model_RGB_to_HSI, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model_HSI_to_RGB = model_HSI_to_RGB.to(device)
        self.model_RGB_to_HSI = model_RGB_to_HSI.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior


    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, mode=None):
        batch_size, c = x.shape[0], condition_x.shape[1]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device).unsqueeze(1).unsqueeze(1)
        # x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if mode == 'HSI_to_RGB':
            x_start, _ = self.model_HSI_to_RGB(torch.cat([condition_x, x], dim=1), timesteps=noise_level)
        elif mode == 'RGB_to_HSI':
            x_start, _ = self.model_RGB_to_HSI(torch.cat([condition_x, x], dim=1), timesteps=noise_level)

        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, img_RGB, img_HSI, t, clip_denoised=True, condition_RGB=None, condition_HSI=None):

        mean1, log_variance1 = self.p_mean_variance(x=img_RGB, t=t, clip_denoised=clip_denoised, condition_x=condition_HSI, mode='HSI_to_RGB')

        mean2, log_variance2 = self.p_mean_variance(x=img_HSI, t=t, clip_denoised=clip_denoised, condition_x=condition_RGB, mode='RGB_to_HSI')

        noise1 = torch.randn_like(img_RGB) if t > 0 else torch.zeros_like(img_RGB)
        noise2 = torch.randn_like(img_HSI) if t > 0 else torch.zeros_like(img_HSI)
        return mean1 + noise1 * (0.5 * log_variance1).exp(), mean2 + noise2 * (0.5 * log_variance2).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, RGB, HSI):
        img_RGB = torch.rand_like(RGB, device=RGB.device)
        img_HSI = torch.rand_like(HSI, device=RGB.device)
        for i in reversed(range(0, self.num_timesteps)):

            img_RGB, img_HSI = self.p_sample(img_RGB, img_HSI, i, condition_RGB=RGB, condition_HSI=HSI)

        return img_RGB, img_HSI

    # Compute loss to train the model
    def p_losses(self, x_in):
        x_start = x_in
        lr_imgs = transforms.Resize(self.img_size)(transforms.Resize(self.LR_size)(x_in))
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(x_start.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The model predict actual noise added at time step t
        pred_noise = self.model(torch.cat([lr_imgs, x_noisy], dim=1), noise_level=sqrt_alpha)

        return self.loss_func(noise, pred_noise), pred_noise, lr_imgs

    def net(self, RGB, HSI):

        RGB = RGB
        HSI = HSI

        b, c, h, w = RGB.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)).to(RGB.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        noise_RGB = torch.randn_like(RGB).to(RGB.device)
        noise_HSI= torch.randn_like(HSI).to(HSI.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy_RGB = sqrt_alpha * RGB + (1 - sqrt_alpha ** 2).sqrt() * noise_RGB
        x_noise_HSI = sqrt_alpha * HSI + (1 - sqrt_alpha ** 2).sqrt() * noise_HSI
        # The model predict actual noise added at time step t
        RGB_to_HSI_x0, _ = self.model_RGB_to_HSI(torch.cat([RGB, x_noise_HSI], dim=1), timesteps=sqrt_alpha)
        HSI_to_RGB_x0, _ = self.model_HSI_to_RGB(torch.cat([HSI, x_noisy_RGB], dim=1), timesteps=sqrt_alpha)

        loss_1 = self.loss_func(HSI, RGB_to_HSI_x0) / int(b * 194 * h * w)
        loss_2 = self.loss_func(RGB, HSI_to_RGB_x0) / int(b * 3 * h * w)

        return loss_1, loss_2

    def forward(self, RGB, HSI, *args, **kwargs):
        return self.net(RGB, HSI, *args, **kwargs)


# Class to train & test desired model
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
        self.img_size = img_size
        self.LR_size = LR_size

        model_HSI_to_RGB = UViT(27, 1, 197, out_chans=3)
        model_RGB_to_HSI = UViT(27, 1, 197, out_chans=194)

        self.sr3 = Diffusion(model_HSI_to_RGB, model_RGB_to_HSI, device, img_size, LR_size, out_channel)
        # Apply weight initialization & set loss & set noise schedule
        self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

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

        train = True

        for i in range(epoch):
            i = i+139
            train_loss = 0
            self.sr3.train()
            randn1 = np.random.randint(0, 100)
            loss_1_epoch = 0
            loss_2_epoch = 0

            if train:
                for step, [RGB, HSI, GT] in enumerate(tqdm(self.dataloader)):
                    # 高光谱和全色图像
                    RGB = RGB.to(self.device).type(torch.float32)
                    HSI = HSI.to(self.device).type(torch.float32)
                    GT = GT.to(self.device).type(torch.float32)
                    b, c, h, w = RGB.shape
                    self.optimizer.zero_grad()
                    loss_1, loss_2 = self.sr3(RGB, HSI)

                    loss_1 = loss_1.sum()
                    loss_2 = loss_2.sum()
                    loss = loss_1 + loss_2

                    loss.backward()
                    self.optimizer.step()

                    loss_1_epoch = loss_1.item() + loss_1_epoch
                    loss_2_epoch = loss_2.item() + loss_2_epoch

                    train_loss += loss.item()

                print('epoch: {}'.format(i))
                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                x.add_column("loss_1", [loss_1_epoch / float(len(self.dataloader))])
                x.add_column("loss_2", [loss_2_epoch / float(len(self.dataloader))])

                print(x)

            if (i + 1) % verbose == 0:
                self.sr3.eval()
                for step, [RGB, HSI, GT] in enumerate(tqdm(self.testloader)):
                    # [RGB, SAR] = test_data
                    RGB = RGB.to(self.device).type(torch.float32)
                    HSI = HSI.to(self.device).type(torch.float32)

                    b, c, h, w = RGB.shape
                    randn3 = np.random.randint(0, b)
                    RGB = RGB[randn3]
                    HSI = HSI[randn3]
                    # Transform to low-resolution images
                    # Save example of test images to check training
                    plt.figure(figsize=(15, 10))
                    plt.subplot(2, 2, 1)
                    plt.axis("off")
                    plt.title("RGB")
                    plt.imshow(np.transpose(torchvision.utils.make_grid(RGB.cpu(),
                                                                        nrow=2, padding=1, normalize=True),(1, 2, 0)))

                    plt.subplot(2, 2, 2)
                    plt.axis("off")
                    plt.title("HSI")
                    # A = self.test(test_img, test_lrHS_img)
                    plt.imshow(np.transpose(torchvision.utils.make_grid(HSI.cpu(),
                                                                        nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :, [50, 100, 150]])

                    plt.subplot(2, 2, 3)
                    plt.axis("off")
                    plt.title("RGB-HSI")
                    self.save(self.save_path, i)
                    print('模型保存成功')
                    img_RGB, img_HSI = self.test(RGB.unsqueeze(0), HSI.unsqueeze(0))
                    plt.imshow(np.transpose(torchvision.utils.make_grid(img_RGB.cpu(),
                                                                        nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :, :])

                    plt.subplot(2, 2, 4)
                    plt.axis("off")
                    plt.title("HSI-RGB")
                    # A = self.test(test_img, test_lrHS_img)
                    plt.imshow(np.transpose(torchvision.utils.make_grid(img_HSI.cpu(),
                                                                        nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :, [50, 100, 150]])

                    plt.savefig('../img/Data_4/feature_5/Result_test_' + str(step) + '.jpg')
                    plt.show()
                    plt.close()
                    break

                # Save model weight

    def test(self, RGB, HSI):

        RGB = RGB
        HSI = HSI

        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(RGB, HSI)
            else:
                result_SR = self.sr3.super_resolution(RGB, HSI)
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


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = 32
    LR_size = 32
    img_size = 128
    root = args.data_path

    train_data = Dataset(root+"T1_RGB.mat", root+"T2_HSI.mat", root+"REF.mat", data=args.data, mode='train', channel=3)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = Dataset(root+"T1_RGB.mat", root+"T2_HSI.mat", root+"REF.mat", data=args.data, mode='test', channel=3)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if cuda else "cpu")
    schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1',
              dataloader=train_dataloader, testloader=test_dataloader, schedule_opt=schedule_opt,
              save_path=args.save_path,
              load_path=args.load_path, load=True,
              inner_channel=64,
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-4, distributed=False)
    sr3.train(epoch=1000, verbose=20)


# from tensorboardX import SummaryWriter
# tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
# tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
