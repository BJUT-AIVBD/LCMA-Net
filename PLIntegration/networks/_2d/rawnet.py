#!usr/bin/env python
# -*- coding:utf-8 _*-

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# region    # <--- layers ---> #

class NonLocal(nn.Module):
    """
    input: (batch, channels, time)
    """

    def __init__(self, input_dim, output_dim):
        super(NonLocal, self).__init__()
        self.theta = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        self.phi = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        self.gt = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        self.ht = nn.Conv1d(in_channels=output_dim, out_channels=input_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, inpt):
        theta = self.theta(inpt)
        phi = self.phi(inpt)
        g = self.gt(inpt)
        w = torch.matmul(torch.transpose(theta, 1, 2), phi)
        w = F.softmax(w, dim=-1)
        w = torch.matmul(g, w)
        w = self.ht(w)
        w = self.bn(w)
        return inpt + w


class FRM(nn.Module):
    def __init__(self, nb_dim, do_add=True, do_mul=True):
        super(FRM, self).__init__()
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()
        self.do_add = do_add
        self.do_mul = do_mul

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        if self.do_mul: x = x * y
        if self.do_add: x = x + y
        return x


class Residual_block_wFRM(nn.Module):
    def __init__(self, nb_filts, first=False, do_mp=True):
        super(Residual_block_wFRM, self).__init__()
        self.first = first
        self.do_mp = do_mp
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        self.frm = FRM(
            nb_dim=nb_filts[1],
            do_add=True,
            do_mul=True)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        if self.do_mp:
            out = self.mp(out)
        out = self.frm(out)
        return out


class Residual_block_NL(nn.Module):
    def __init__(self, nb_filts, first=False, do_mp=True, activate=None):
        super(Residual_block_NL, self).__init__()
        self.first = first
        self.do_mp = do_mp
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        self.sa = NonLocal(input_dim=nb_filts[2],
                           output_dim=nb_filts[3])

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        if self.do_mp:
            out = self.mp(out)
        out = self.sa(out)
        return out


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50, device="cuda:0"):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size).to(device)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate).to(device)
        # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        # self.n_ = self.n_.to(waveforms.device)
        #
        # self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


# endregion # <--- layers ---> #

class RawNet_NL_GRU(nn.Module):
    def __init__(self, nl_idim=256, nl_odim=64, **kwargs):
        super().__init__()
        # region    # <--- module arch ---> #
        self.ln = LayerNorm(59049)
        self.first_conv = SincConv_fast(in_channels=1,
                                        out_channels=128,
                                        kernel_size=251)

        self.first_bn = nn.BatchNorm1d(num_features=128)
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.block0 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 128], first=True))
        self.block1 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 128]))
        self.block2 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 256]))
        self.block3 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, nl_idim, nl_odim]))
        self.block4 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, nl_idim, nl_odim]))
        self.block5 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, nl_idim, nl_odim]))

        self.bn_before_gru = nn.BatchNorm1d(num_features=256)
        self.gru = nn.GRU(input_size=256,
                          hidden_size=1024,
                          num_layers=1,
                          bidirectional=False,
                          batch_first=True)
        self.fc1_gru = nn.Linear(in_features=1024,
                                 out_features=1024)
        self.sig = nn.Sigmoid()
        # endregion # <--- module arch ---> #

    def forward(self, x):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.ln(x)
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)

        x = self.block0(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        code = self.fc1_gru(x)

        return code

    def load_from_checkpoint(self, fp, args=None, **kwargs):
        state_dict = torch.load(fp)["state_dict"]
        self.load_state_dict(state_dict, strict=False)
