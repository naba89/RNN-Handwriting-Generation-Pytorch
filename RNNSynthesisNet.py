import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
eps = float(np.finfo(np.float32).eps)


class WindowLayer(nn.Module):
    def __init__(self, hidden_size, num_mixture):
        super(WindowLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_mixture = num_mixture
        self.linear = nn.Linear(hidden_size, 3 * num_mixture)
        self.kappa_prev = None

    def forward(self, x, chars):
        char_len = chars.size()[1]
        # x = x.view(-1, self.hidden_size)
        out = self.linear(x)
        alpha, beta, kappa = torch.chunk(out, 3, dim=1)
        alpha = torch.unsqueeze(torch.exp(alpha), dim=2)
        beta = torch.unsqueeze(torch.exp(beta), dim=2)
        kappa = torch.unsqueeze(kappa, dim=2)

        if self.kappa_prev is None:
            self.kappa_prev = torch.zeros_like(kappa)

        kappa = self.kappa_prev + torch.exp(kappa)
        self.kappa_prev = Variable(kappa.data)

        if USE_CUDA:
            u_ = Variable(torch.from_numpy(np.arange(char_len, dtype=np.float32)).cuda().view(1, 1, -1))
        else:
            u_ = Variable(torch.from_numpy(np.arange(char_len, dtype=np.float32)).view(1, 1, -1))

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u_) ** 2), dim=1, keepdim=True)
        window = torch.squeeze(torch.bmm(phi, chars), dim=1)

        return window


class RNNSynthesisNet(nn.Module):

    def __init__(self, args):
        super(RNNSynthesisNet, self).__init__()

        # LSTM settings
        self.args = args
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.sequence_length = args.seq_length
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.c_dimension = args.c_dimension

        # MDN settings
        self.num_mixture = args.num_mixture
        self.out_size = 1 + self.num_mixture * 6

        # LSTM layers
        self.lstm1 = nn.LSTMCell(self.input_size + self.c_dimension, self.hidden_size, bias=True)
        self.windowlayer = WindowLayer(self.hidden_size, self.num_mixture)
        # self.lstm2 = nn.LSTMCell(self.input_size + self.c_dimension + self.hidden_size, self.hidden_size, bias=True)
        self.lstm2 = nn.LSTMCell(self.input_size + self.c_dimension, self.hidden_size, bias=True)

        # MDN layers
        self.MDN = nn.Linear(in_features=self.hidden_size, out_features=self.out_size)

        # Initialize
        self.hidden1, self.window, self.hidden2, self.kappa = self.initialize()

    def initialize(self):

        if USE_CUDA:
            hidden1_init = (Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                            Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()))
            window_init = Variable(torch.zeros(self.batch_size, self.c_dimension).cuda())
            hidden2_init = (Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                            Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()))
            kappa_init = Variable(torch.zeros(self.batch_size, self.num_mixture).cuda())
        else:
            hidden1_init = (Variable(torch.zeros(self.batch_size, self.hidden_size)),
                            Variable(torch.zeros(self.batch_size, self.hidden_size)))
            window_init = Variable(torch.zeros(self.batch_size, self.c_dimension))
            hidden2_init = (Variable(torch.zeros(self.batch_size, self.hidden_size)),
                            Variable(torch.zeros(self.batch_size, self.hidden_size)))
            kappa_init = Variable(torch.zeros(self.batch_size, self.num_mixture))

        return hidden1_init, window_init, hidden2_init, kappa_init

    def repackage_hidden(self, h):
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, x, chars):
        self.hidden1 = self.repackage_hidden(self.hidden1)
        self.hidden2 = self.repackage_hidden(self.hidden2)
        self.window = self.repackage_hidden(self.window)
        self.kappa = self.repackage_hidden(self.kappa)
        b, s, _ = x.size()
        outputs = []
        for i in range(s):
            points = x[:, i, :]
            # print(points.size(), self.window.size(), self.hidden1[0].size())
            # if not self.training:
            #     print(points.size(), self.window.size())
            inp1 = torch.cat([points, self.window], dim=1)
            self.hidden1 = self.lstm1(inp1, self.hidden1)
            self.window = self.windowlayer(self.hidden1[0], chars)
            # print(self.window.size(), inp.size())
            inp2 = torch.cat([points, self.window], dim=1)
            self.hidden2 = self.lstm2(inp2, self.hidden2)
            outputs.append(self.hidden2[0])
        outputs = torch.stack(outputs, dim=0)
        # print(outputs.size())
        outputs = outputs.transpose(0, 1).contiguous()
        outputs = outputs.view(-1, self.hidden_size)

        mdn_out = self.MDN(outputs)

        return self.split_mdn_out(mdn_out)

    @staticmethod
    def split_mdn_out(x):
        eos = F.sigmoid(x[:, 0]).view(-1, 1).contiguous()

        pi, mu1, mu2, sigma1, sigma2, rho = torch.chunk(tensor=x[:, 1:], chunks=6, dim=1)

        pi = F.softmax(pi, dim=1)
        sigma1 = torch.exp(sigma1)
        sigma2 = torch.exp(sigma2)
        rho = F.tanh(rho)

        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def sample(self, num_samples=1200, chars=None):
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

        if chars is None:
            return None

        self.eval()
        prev_x = Variable(torch.zeros(1, 1, self.input_size).cuda()) if USE_CUDA \
            else Variable(torch.zeros(1, 1, self.input_size))
        prev_x[0, 0, 2] = 1.
        strokes = np.zeros((num_samples, 3), dtype=np.float32)

        mixture_params = []

        for i in range(num_samples):
            eos, pi, mu1, mu2, sigma1, sigma2, rho = self.forward(prev_x, chars)

            o_pi = pi.data.cpu().numpy()[0] if USE_CUDA else pi.data.numpy()[0]
            idx = get_pi_idx(random.random(), o_pi)

            o_eos = 1 if random.random() < eos.data[0][0] else 0

            next_x1, next_x2 = sample_gaussian_2d(
                mu1.data[0][idx], mu2.data[0][idx], sigma1.data[0][idx], sigma2.data[0][idx], rho.data[0][idx])

            strokes[i, :] = [next_x1, next_x2, o_eos]

            if USE_CUDA:
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

            prev_x = Variable(torch.zeros(1, 1, self.input_size).cuda()) if USE_CUDA \
                else Variable(torch.zeros(1, 1, self.input_size))

            prev_x[0, 0, 0] = next_x1
            prev_x[0, 0, 1] = next_x2
            prev_x[0, 0, 2] = o_eos

        strokes[:, 0:2] *= self.args.data_scale
        return strokes, mixture_params
