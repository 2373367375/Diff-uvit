from typing import List

import torch
from torch import nn
from torch.nn.functional import l1_loss, cross_entropy
from hyperparams import *

class AceNet(nn.Module):

    def __init__(self):
        super().__init__()
        # self.automatic_optimization = True
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.channel_hsi = 3

        self.Rx = nn.Sequential(
            nn.Conv2d(self.channel_hsi, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.Py = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(100, 50, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(50, 20, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.Sz = nn.Sequential(
            nn.ConvTranspose2d(20, 50, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(50, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(100, 3, kernel_size=(3, 3)),
        )

        self.Qz = nn.Sequential(
            nn.ConvTranspose2d(20, 50, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(50, 100, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(100, self.channel_hsi, kernel_size=(3, 3)),
        )

        # self.discriminator = nn.Sequential(
        #     nn.Conv2d(20, 64, 3,1,1),
        #     nn.LeakyReLU(negative_slope=0.3),
        #     nn.Conv2d(64, 32, 3,1,1),
        #     nn.LeakyReLU(negative_slope=0.3),
        #     nn.Conv2d(32, 16, 3,1,1),
        #     nn.Flatten(),
        #     nn.Linear(16*21*21, 1),
        #     nn.Sigmoid()
        # ).to('cuda:0')
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(27*27,2)

    def forward(self, x, y, prior_info):
        x = x[:,[40,80,120],:,:]
        y_hat = self.Sz(self.Rx(x))
        x_translation_loss = l1_loss(y, y_hat)
        x_cycled = self.Qz(self.Py(y_hat))
        x_cycle_loss = l1_loss(x, x_cycled)

        x_reconstructed = self.Qz(self.Rx(x))
        x_recon_loss = l1_loss(x, x_reconstructed)
        x_hat = self.Qz(self.Py(y))
        y_translation_loss = l1_loss(x, x_hat)
        y_cycled = self.Sz(self.Rx(x_hat))
        y_cycle_loss = l1_loss(y, y_cycled)
        y_reconstructed = self.Sz(self.Py(y))
        y_recon_loss = l1_loss(y, y_reconstructed)

        W_RECON = 1
        W_CYCLE = 1
        W_D = 1
        W_HAT = 1
        total_AE_loss = (
                W_RECON * (x_recon_loss + y_recon_loss) +
                W_HAT * (x_translation_loss + y_translation_loss) +
                W_CYCLE * (x_cycle_loss + y_cycle_loss)
        )

        out_image_t1 = torch.sum((x - x_hat) ** 2, dim=1)
        out_image_t1 = out_image_t1 / torch.max(out_image_t1.reshape(out_image_t1.size(0),-1), dim=1)[0].unsqueeze(1).unsqueeze(2).repeat(1,27,27)
        out_image_t2 = torch.sum((y - y_hat) ** 2, dim=1)
        out_image_t2 = out_image_t2 / torch.max(out_image_t2.reshape(out_image_t1.size(0),-1), dim=1)[0].unsqueeze(1).unsqueeze(2).repeat(1,27,27)
        diff_patch = (out_image_t1 + out_image_t2) / 2.0
        diff_patch = self.flatten(diff_patch)
        output = self.linear(diff_patch)

        # generator_code = self.discriminator(torch.cat((self.Rx(x), self.Py(y))))
        # x_disc, y_disc = torch.tensor_split(generator_code, 2, dim=0)
        # disc_code_loss = W_D * (mse_loss(torch.zeros_like(x_disc), x_disc) + mse_loss(torch.ones_like(y_disc), y_disc))
        #
        # disc_out = self.discriminator(torch.cat((self.Rx(x), self.Py(y))))
        # x_disc, y_disc = torch.tensor_split(disc_out, 2, dim=0)
        # disc_loss = W_D * (mse_loss(torch.ones_like(x_disc), x_disc) + mse_loss(torch.zeros_like(y_disc), y_disc))

        return output
        # return total_AE_loss, output

    def training_step(self, test_batch, batch_idx):
        x, y, prior_info = test_batch
        # x = x.permute(0, 3, 1, 2).type(torch.float)
        # y = y.permute(0, 3, 1, 2).type(torch.float)
        x = x.type(torch.float)
        y = y.type(torch.float)

        opt_encoders_decoders, opt_encoders, opt_disc = self.configure_optimizers()

        opt_encoders_decoders.zero_grad()

        # X data flow
        y_hat = self.Sz(self.Rx(x))
        x_translation_loss = self.mse_loss_weighted(y, y_hat, 1 - prior_info)
        x_cycled = self.Qz(self.Py(y_hat))
        x_cycle_loss = mse_loss(x, x_cycled)
        x_reconstructed = self.Qz(self.Rx(x))
        x_recon_loss = mse_loss(x, x_reconstructed)

        # Y data flow
        x_hat = self.Qz(self.Py(y))
        y_translation_loss = self.mse_loss_weighted(x, x_hat, 1 - prior_info)
        y_cycled = self.Sz(self.Rx(x_hat))
        y_cycle_loss = mse_loss(y, y_cycled)
        x_hat = self.Sz(self.Py(y))
        y_recon_loss = mse_loss(y, x_hat)

        total_AE_loss = (
                W_RECON * (x_recon_loss + y_recon_loss) +
                W_HAT * (x_translation_loss + y_translation_loss) +
                W_CYCLE * (x_cycle_loss + y_cycle_loss)
        )

        self.log("Reconstruction loss", W_RECON * (x_recon_loss + y_recon_loss))
        self.log("Prior information loss", W_HAT * (x_translation_loss + y_translation_loss))
        self.log("Total AutoEncoders loss", total_AE_loss, prog_bar=True)

        self.manual_backward(total_AE_loss)
        opt_encoders_decoders.step()

        opt_encoders.zero_grad()

        generator_code = self.discriminator(torch.cat((self.Rx(x), self.Py(y))))
        x_disc, y_disc = torch.tensor_split(generator_code, 2, dim=0)
        disc_code_loss = W_D * (mse_loss(torch.zeros_like(x_disc), x_disc) + mse_loss(torch.ones_like(y_disc), y_disc))
        self.log("Discriminator code loss", disc_code_loss)

        self.manual_backward(disc_code_loss)
        opt_encoders.step()

        opt_disc.zero_grad()

        disc_out = self.discriminator(torch.cat((self.Rx(x), self.Py(y))))
        x_disc, y_disc = torch.tensor_split(disc_out, 2, dim=0)

        disc_loss = W_D * (mse_loss(torch.ones_like(x_disc), x_disc) + mse_loss(torch.zeros_like(y_disc), y_disc))

        self.log("Discriminator loss", disc_loss)

        self.manual_backward(disc_loss)
        opt_disc.step()

    def configure_optimizers(self) -> List[torch.optim.Adam]:
        encoders_params = list(self.Rx.parameters()) + list(self.Py.parameters())
        decoders_params = list(self.Sz.parameters()) + list(self.Qz.parameters())

        optimizer_ae = torch.optim.Adam(encoders_params + decoders_params, lr=self.learning_rate,
                                        weight_decay=W_REG)
        optimizer_e = torch.optim.Adam(encoders_params, lr=self.learning_rate, weight_decay=W_REG)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                          weight_decay=W_REG)

        return [optimizer_ae, optimizer_e, optimizer_disc]

    @staticmethod
    def mse_loss_weighted(x, x_hat, weights):
        L2_Norm = torch.linalg.norm(x_hat - x, dim=1) ** 2
        weighted_L2_norm: torch.Tensor = L2_Norm * (weights.unsqueeze(1).unsqueeze(1).repeat(1,27,27))
        loss = weighted_L2_norm.mean()
        return loss

    @property
    def automatic_optimization(self) -> bool:
        return False