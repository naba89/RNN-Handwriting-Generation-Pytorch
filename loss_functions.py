import math

import torch.nn as nn
import torch
import numpy as np

eps = float(np.finfo(np.float32).eps)


class PredictionLoss(nn.Module):
    def __init__(self, batch_size, seq_len):
        super(PredictionLoss, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

    @staticmethod
    def gaussian_2d(x1, x2, mu1, mu2, s1, s2, rho):

        norm1 = x1 - mu1
        norm2 = x2 - mu2

        sigma1sigma2 = s1 * s2

        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2

        numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        gaussian = numerator / denominator

        return gaussian

    def forward(self, output, target):

        eos, pi, mu1, mu2, sigma1, sigma2, rho = output

        x_1, x_2, x_eos = torch.chunk(target.view(-1, 3).contiguous(), chunks=3, dim=2)

        gaussian = self.gaussian_2d(x_1, x_2, mu1, mu2, sigma1, sigma2, rho)

        loss_gaussian = -torch.log(torch.sum(pi * gaussian, dim=1, keepdim=True) + eps)

        loss_bernoulli = -torch.log(eos * x_eos + (1 - eos) * (1 - x_eos))

        loss = torch.sum(loss_gaussian + loss_bernoulli)

        return loss / (self.batch_size * self.seq_len)

