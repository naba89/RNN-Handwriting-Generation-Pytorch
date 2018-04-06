import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
eps = float(np.finfo(np.float32).eps)


class RNNPredictNet(nn.Module):

    def __init__(self, args):
        super(RNNPredictNet, self).__init__()

        # LSTM settings
        self.args = args
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.num_directions = 2 if args.bidirectional else 1
        self.sequence_length = args.seq_length
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size

        # MDN settings
        self.num_mixture = args.num_mixture
        self.out_size = 1 + self.num_mixture * 6

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=1 - args.keep_prob)
        self.hidden = None

        # MDN layers
        self.MDN = nn.Linear(in_features=self.hidden_size, out_features=self.out_size)

    @staticmethod
    def split_mdn_out(x):
        eos = F.sigmoid(x[:, 0]).view(-1, 1).contiguous()

        pi, mu1, mu2, sigma1, sigma2, rho = torch.chunk(tensor=x[:, 1:], chunks=6, dim=1)

        pi = F.softmax(pi, dim=1)
        sigma1 = torch.exp(sigma1)
        sigma2 = torch.exp(sigma2)
        rho = F.tanh(rho)

        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()

        if self.hidden is not None:
            self.hidden = self.repackage_hidden(self.hidden)
            lstms_out, self.hidden = self.lstm(x, self.hidden)
        else:
            lstms_out, self.hidden = self.lstm(x)

        lstms_out = lstms_out.transpose(0, 1).contiguous()
        lstms_out = lstms_out.view(-1, self.hidden_size)

        mdn_out = self.MDN(lstms_out)

        return self.split_mdn_out(mdn_out)

    def sample(self, num_samples=1200):
        def get_pi_idx(x, pdf):
            n = pdf.size
            accumulate = 0
            for ii in range(0, n):
                accumulate += pdf[ii]
                if accumulate >= x:
                    return ii
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu_1, mu_2, s1, s2, rho_):
            mean = [mu_1, mu_2]
            cov = [[s1 * s1, rho_ * s1 * s2], [rho_ * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mean, cov, 1)

            return x[0][0], x[0][1]

        self.eval()
        prev_x = Variable(torch.zeros(1, 1, self.input_size).cuda()) if use_cuda \
            else Variable(torch.zeros(1, 1, self.input_size))
        prev_x[0, 0, 2] = 1.
        strokes = np.zeros((num_samples, 3), dtype=np.float32)

        self.hidden = None

        mixture_params = []

        for i in range(num_samples):
            eos, pi, mu1, mu2, sigma1, sigma2, rho = self.forward(prev_x)

            o_pi = pi.data.cpu().numpy()[0] if use_cuda else pi.data.numpy()[0]
            idx = get_pi_idx(random.random(), o_pi)

            o_eos = 1 if random.random() < eos.data[0][0] else 0

            next_x1, next_x2 = sample_gaussian_2d(
                mu1.data[0][idx], mu2.data[0][idx], sigma1.data[0][idx], sigma2.data[0][idx], rho.data[0][idx])

            strokes[i, :] = [next_x1, next_x2, o_eos]

            if use_cuda:
                params = [
                    pi.data.cpu().numpy()[0],
                    mu1.data.cpu().numpy()[0],
                    mu2.data.cpu().numpy()[0],
                    sigma1.data.cpu().numpy()[0],
                    sigma2.data.cpu().numpy()[0],
                    rho.data.cpu().numpy()[0],
                    eos.data.cpu().numpy()[0]]
            else:
                params = [
                    pi.data.numpy()[0],
                    mu1.data.numpy()[0],
                    mu2.data.numpy()[0],
                    sigma1.data.numpy()[0],
                    sigma2.data.numpy()[0],
                    rho.data.numpy()[0],
                    eos.data.numpy()[0]]
            mixture_params.append(params)

            prev_x = Variable(torch.zeros(1, 1, self.input_size).cuda()) if use_cuda \
                else Variable(torch.zeros(1, 1, self.input_size))

            prev_x[0, 0, 0] = next_x1
            prev_x[0, 0, 1] = next_x2
            prev_x[0, 0, 2] = o_eos

        strokes[:, 0:2] *= self.args.data_scale
        return strokes, mixture_params
