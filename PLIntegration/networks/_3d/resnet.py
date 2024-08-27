#!usr/bin/env python
# -*- coding:utf-8 _*-
from functools import partial

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


# region    # <--- ResNeXt3D ---> #

def conv3x3x3(in_planes, out_planes, stride=(3, 3, 3)):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(3, 3, 3),
                     stride=stride,
                     padding=(1, 1, 1),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(1, 1, 1),
                     stride=stride,
                     bias=False)


class ResNeXtBottleneck3D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=(1, 1, 1),
                 downsample=None, input_shape=None):
        super().__init__()
        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes,
                               mid_planes,
                               kernel_size=(3, 3, 3),
                               stride=stride,
                               padding=(1, 1, 1),
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class STDABottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=(1, 1, 1),
                 downsample=None, input_shape=None):
        super().__init__()
        mid_planes = cardinality * planes // 32
        input_shape[0] = mid_planes
        self.conv1 = conv1x1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = STDA(input_shape,
                          mid_planes,
                          kernel_size=3,
                          padding=1,
                          stride=stride,
                          groups=cardinality,
                          bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


# def generate_model(model_depth, **kwargs):
#     model = None
#     assert model_depth in [50, 101, 152, 200]
#
#     if model_depth == 50:
#         model = ResNeXt3D(ResNeXtBottleneck3D, [3, 4, 6, 3], get_inplanes(),
#                           **kwargs)
#     elif model_depth == 101:
#         model = ResNeXt3D(ResNeXtBottleneck3D, [3, 4, 23, 3], get_inplanes(),
#                           **kwargs)
#     elif model_depth == 152:
#         model = ResNeXt3D(ResNeXtBottleneck3D, [3, 8, 36, 3], get_inplanes(),
#                           **kwargs)
#     elif model_depth == 200:
#         model = ResNeXt3D(ResNeXtBottleneck3D, [3, 24, 36, 3], get_inplanes(),
#                           **kwargs)
#
#     return model

# endregion # <--- ResNeXt3D ---> #

# region    # <--- Deformable Convolution ---> #

class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters * 2, (3, 3), padding=(1, 1), bias=False,
                                           **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets_ = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        # 这个self._to_bc_h_w_2就是我修改的代码
        offsets = self._to_bc_h_w_2(offsets_, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        X_offset = self.th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(X_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.shape
        dtype, cuda = x.data.type(), x.data.is_cuda
        # if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
        #     return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = self.th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, x_shape[2], x_shape[3], 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, x_shape[2], x_shape[3])
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, x_shape[1], x_shape[2], x_shape[3])
        return x

    def th_generate_grid(self, batch_size, input_height, input_width, dtype, cuda):
        grid = torch.meshgrid(torch.arange(0, input_height), torch.arange(0, input_width))
        grid = torch.stack(grid, dim=-1)
        grid = grid.view(-1, 2)
        grid = grid.repeat(batch_size, 1, 1)
        # grid = self.np_repeat_2d(grid, batch_size)
        # grid = torch.from_numpy(grid).type(dtype)
        if cuda:
            grid = grid.cuda()
        # return torch.autograd.Variable(grid, requires_grad=False)
        return grid

    def th_batch_map_offsets(self, input, offsets, grid=None, order=1):
        """Batch map offsets into input
        Parameters
        ---------
        input : torch.Tensor. shape = (b*c, s, s)
        offsets: torch.Tensor. shape = (b*c, s, s, 2)
        Returns
        -------
        torch.Tensor. shape = (b*c, s, s)
        """
        batch_size = input.size(0)
        input_height = input.size(1)
        input_width = input.size(2)

        # offsets: (b*c, h*w, 2)
        offsets = offsets.view(batch_size, -1, 2)
        if grid is None:
            # grid: (b*c, h*w, 2)
            grid = self.th_generate_grid(batch_size, input_height, input_width, offsets.data.type(),
                                         offsets.data.is_cuda)
        # coords: (b*c, h*w, 2)
        coords = offsets + grid

        mapped_vals = self.th_batch_map_coordinates(input, coords)
        return mapped_vals

    def np_repeat_2d(self, a, repeats):
        """Tensorflow version of np.repeat for 2D"""

        assert len(a.shape) == 2
        a = np.expand_dims(a, 0)
        a = np.tile(a, [repeats, 1, 1])
        return a

    def th_batch_map_coordinates(self, input, coords, order=1):
        """Batch version of th_map_coordinates
        Only supports 2D feature maps
        Parameters
        ----------
        input : tf.Tensor. shape = (b*c, s, s)
        coords : tf.Tensor. shape = (b*c, s*s, 2)
        Returns
        -------
        tf.Tensor. shape = (b*c, s, s)
        """

        batch_size = input.size(0)
        input_height = input.size(1)
        input_width = input.size(2)

        n_coords = coords.size(1)

        # coords = torch.clamp(coords, 0, input_size - 1)
        # 限制坐标的取值范围
        coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1),
                            torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

        # assert (coords.size(1) == n_coords)

        # l:left, r:right, t:top, b:below
        coords_lt = coords.floor().long().float()
        coords_rb = coords.ceil().long().float()
        coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
        coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
        idx = self.th_repeat(torch.arange(0, batch_size), n_coords)
        idx = torch.autograd.Variable(idx, requires_grad=False)
        if input.is_cuda:
            idx = idx.cuda()

        def _get_vals_by_coords(input, coords):
            indices = torch.stack([
                idx, self.th_flatten(coords[..., 0]), self.th_flatten(coords[..., 1])
            ], 1)
            inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
            vals = self.th_flatten(input).index_select(0, inds.detach().long())
            vals = vals.view(batch_size, n_coords)
            return vals

        vals_lt = _get_vals_by_coords(input, coords_lt.detach())
        vals_rb = _get_vals_by_coords(input, coords_rb.detach())
        vals_lb = _get_vals_by_coords(input, coords_lb.detach())
        vals_rt = _get_vals_by_coords(input, coords_rt.detach())

        coords_offset_lt = coords - coords_lt.type(coords.data.type())
        vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
        vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
        mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
        return mapped_vals

    def th_repeat(self, a, repeats, axis=0):
        """Torch version of np.repeat for 1D"""
        assert len(a.size()) == 1
        return self.th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))

    def th_flatten(self, a):
        """Flatten tensor"""
        return a.contiguous().view(a.nelement())


class DeformConv2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, **kwargs):
        super(DeformConv2D, self).__init__()

        self.offset = ConvOffset2D(in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride,
                              **kwargs)

    def forward(self, x):
        x = self.offset(x)
        x = self.conv(x)

        return x

    def parameters(self, **kwargs):
        return filter(lambda p: p.requires_grad, super(DeformConv2D, self).parameters())


class ConvOffset3D(nn.Conv3d):
    """ConvOffset3D

    Convolutional layer responsible for learning the 3D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con3d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset3D, self).__init__(self.filters, self.filters * 3, (3, 3, 3), padding=(1, 1, 1), bias=False,
                                           **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets_ = super(ConvOffset3D, self).forward(x)

        # offsets: (b*c, l, h, w, 3)
        offsets = self._to_bc_l_h_w_3(offsets_, x_shape)

        # x: (b*c, l, h, w)
        x = self._to_bc_l_h_w(x, x_shape)

        # X_offset: (b*c, l, h, w)
        x_offset = self.th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))

        # x_offset: (b, c, l, h, w)
        x_offset = self._to_b_c_l_h_w(x_offset, x_shape)

        return x_offset
        # return x_offset, offsets_

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_length, input_height, input_width = x.size(0), x.size(1), x.size(2), x.size(3)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_length, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_length, input_height, input_width, dtype, cuda)
        self._grid = self.th_generate_grid(batch_size, input_length, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3) * weights.size(4)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_l_h_w_3(x, x_shape):
        """(b, 3c, l, h, w) -> (b*c, l, h, w, 3)"""
        return x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]), 3)

    @staticmethod
    def _to_bc_l_h_w(x, x_shape):
        """(b, c, l, h, w) -> (b*c, l, h, w)"""
        return x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))

    @staticmethod
    def _to_b_c_l_h_w(x, x_shape):
        """(b*c, l, h, w) -> (b, c, l, h, w)"""
        return x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))

    def th_generate_grid(self, batch_size, input_length, input_height, input_width, dtype, cuda):
        grid = np.meshgrid(
            range(input_length), range(input_height), range(input_width), indexing='ij'
        )
        grid = np.stack(grid, axis=-1)
        grid = grid.reshape(-1, 3)

        grid = self.np_repeat_3d(grid, batch_size)
        grid = torch.from_numpy(grid).type(dtype).squeeze()
        if cuda:
            grid = grid.cuda()
        return torch.autograd.Variable(grid, requires_grad=False)

    def th_batch_map_offsets(self, input, offsets, grid=None, order=1):
        """Batch map offsets into input
        Parameters
        ---------
        input : torch.Tensor. shape = (b*c, s, s, s)
        offsets: torch.Tensor. shape = (b*c, s, s, s, 3)
        Returns
        -------
        torch.Tensor. shape = (b*c, s, s, s)
        """
        batch_size = input.size(0)
        input_length = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        # offsets: (b*c, l*h*w, 3)
        offsets = offsets.view(batch_size, -1, 3)
        if grid is None:
            # grid: (b*c, l*h*w, 3)
            grid = self.th_generate_grid(batch_size, input_length, input_height, input_width, offsets.data.type(),
                                         offsets.data.is_cuda)
        # coords: (b*c, l*h*w, 3)
        coords = offsets + grid

        mapped_vals = self.th_batch_map_coordinates(input, coords)
        return mapped_vals

    def np_repeat_3d(self, a, repeats):
        """Tensorflow version of np.repeat for 2D"""

        assert len(a.shape) == 2
        a = np.expand_dims(a, 0)
        a = np.tile(a, [repeats, 1, 1, 1])
        return a

    def th_batch_map_coordinates(self, input, coords, order=1):
        """Batch version of th_map_coordinates
        Only supports 2D feature maps
        Parameters
        ----------
        input : tf.Tensor. shape = (b*c, z, y, x)
        coords : tf.Tensor. shape = (b*c, s*s*s, 3)
        Returns
        -------
        tf.Tensor. shape = (b*c, s, s, s)
        """

        batch_size = input.size(0)
        input_length = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        n_coords = coords.size(1)

        # coords = torch.clamp(coords, 0, input_size - 1)
        # 限制坐标的取值范围
        coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_length - 1),
                            torch.clamp(coords.narrow(2, 1, 1), 0, input_height - 1),
                            torch.clamp(coords.narrow(2, 2, 1), 0, input_width - 1)), 2)

        # assert (coords.size(1) == n_coords)

        # reference: https://blog.csdn.net/webzhuce/article/details/86585489
        coords_000 = coords.floor().long().float()
        coords_111 = coords.ceil().long().float()
        coords_100 = torch.stack([coords_000[..., 0], coords_000[..., 1], coords_111[..., 2]], 2)
        coords_010 = torch.stack([coords_000[..., 0], coords_111[..., 1], coords_000[..., 2]], 2)
        coords_110 = torch.stack([coords_000[..., 0], coords_111[..., 1], coords_111[..., 2]], 2)
        coords_001 = torch.stack([coords_111[..., 0], coords_000[..., 1], coords_000[..., 2]], 2)
        coords_101 = torch.stack([coords_111[..., 0], coords_000[..., 1], coords_111[..., 2]], 2)
        coords_011 = torch.stack([coords_111[..., 0], coords_111[..., 1], coords_000[..., 2]], 2)
        idx = self.th_repeat(torch.arange(0, batch_size), n_coords)
        idx = torch.autograd.Variable(idx, requires_grad=False)
        if input.is_cuda:
            idx = idx.cuda()

        def _get_vals_by_coords(input, coords):
            indices = torch.stack([
                idx, self.th_flatten(coords[..., 0]), self.th_flatten(coords[..., 1]), self.th_flatten(coords[..., 2])
            ], 1)
            inds = indices[:, 0] * input.size(1) * input.size(2) * input.size(3) + \
                   indices[:, 1] * input.size(2) * input.size(3) + \
                   indices[:, 2] * input.size(3) + \
                   indices[:, 3]
            vals = self.th_flatten(input).index_select(0, inds.detach().long())
            vals = vals.view(batch_size, n_coords)
            return vals

        vals_000 = _get_vals_by_coords(input, coords_000.detach())
        vals_111 = _get_vals_by_coords(input, coords_111.detach())
        vals_001 = _get_vals_by_coords(input, coords_001.detach())
        vals_010 = _get_vals_by_coords(input, coords_010.detach())
        vals_011 = _get_vals_by_coords(input, coords_011.detach())
        vals_100 = _get_vals_by_coords(input, coords_100.detach())
        vals_101 = _get_vals_by_coords(input, coords_101.detach())
        vals_110 = _get_vals_by_coords(input, coords_110.detach())

        zd, yd, xd = torch.split(coords - coords_000.type(coords.data.type()), 1, 2)
        zd = torch.squeeze(zd)
        yd = torch.squeeze(yd)
        xd = torch.squeeze(xd)
        mapped_vals = vals_000 * (1 - xd) * (1 - yd) * (1 - zd) + \
                      vals_100 * xd * (1 - yd) * (1 - zd) + \
                      vals_010 * (1 - xd) * yd * (1 - zd) + \
                      vals_001 * (1 - xd) * (1 - yd) * zd + \
                      vals_101 * xd * (1 - yd) * zd + \
                      vals_011 * (1 - xd) * yd * zd + \
                      vals_110 * xd * yd * (1 - zd) + \
                      vals_111 * xd * yd * zd
        return mapped_vals

    def th_repeat(self, a, repeats, axis=0):
        """Torch version of np.repeat for 1D"""
        assert len(a.size()) == 1
        return self.th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))

    def th_flatten(self, a):
        """Flatten tensor"""
        return a.contiguous().view(a.nelement())


class DeformConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, **kwargs):
        super(DeformConv3D, self).__init__()

        self.offset = ConvOffset3D(in_channel)
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride,
                              **kwargs)

    def forward(self, x):
        x = self.offset(x)
        x = self.conv(x)

        return x

    def parameters(self, **kwargs):
        return filter(lambda p: p.requires_grad, super(DeformConv3D, self).parameters())


# endregion # <--- Deformable Convolution ---> #

# region    # <--- STDA ---> #

class SpatialAttention(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super(SpatialAttention, self).__init__()
        c, l, h, w = input_shape
        self.theta = nn.Conv2d(c * l, 1, kernel_size=(1, 1), bias=False)
        self.g = nn.Conv2d(c * l, 1, kernel_size=(1, 1), bias=False)
        self.h = nn.Conv2d(1, c * l, kernel_size=(1, 1))
        self.f = nn.AvgPool1d(h * w)

    def forward(self, inpt):
        b, c, l, h, w = inpt.shape
        inpt = inpt.view(b, c * l, h, w)
        t = self.theta(inpt).view(b, 1, -1)
        g = self.g(inpt).view(b, 1, -1).transpose(1, 2)
        x = torch.mul(t, g)
        x = self.f(x)
        x = x.view(b, 1, h, w)
        x = self.h(x)
        x = inpt + x
        return x.view(b, c * l, h, w)


class TemporalAttention(nn.Module):
    def __init__(self, input_shape):
        super(TemporalAttention, self).__init__()
        c, l, h, w = input_shape
        self.theta = nn.Conv3d(c, 1, kernel_size=(1, 1, 1), bias=False)
        self.g = nn.Conv3d(c, 1, kernel_size=(1, 1, 1), bias=False)
        self.h = nn.Conv3d(1, c, kernel_size=(1, 1, 1))
        self.f = nn.AvgPool1d(l * h * w)

    def forward(self, inpt):
        b, c, l, h, w = inpt.shape
        t = self.theta(inpt).view(b, 1, -1)
        g = self.g(inpt).view(b, 1, -1).transpose(1, 2)
        x = torch.mul(t, g)
        x = self.f(x)
        x = x.view(b, 1, l, h, w)
        x = self.h(x)
        return inpt + x


class STDA(nn.Module):
    """
    input_shape = [1024, 2, 7, 7]
    out_channel = 1024 or 2048
    kernel_size = 3
    padding = 1
    stride = 1
    """

    def __init__(self, input_shape, out_channel, kernel_size, padding, stride, alpha=0.5, **kwargs):
        super(STDA, self).__init__()
        self.spt_att = SpatialAttention(input_shape)
        self.tep_att = TemporalAttention(input_shape)
        self.df_conv3d = DeformConv3D(input_shape[0], out_channel, kernel_size, padding, stride=1, **kwargs)
        self.df_conv2d = DeformConv2D(input_shape[0] * input_shape[1], out_channel * input_shape[1], kernel_size,
                                      padding, stride=1, **kwargs)
        self.conv3d_1 = nn.Conv3d(out_channel, out_channel, kernel_size, stride, padding, **kwargs)
        self.conv3d_2 = nn.Conv3d(out_channel, out_channel, kernel_size, stride, padding, **kwargs)
        self.alpha = alpha

    def forward(self, inpt):
        yt = self.tep_att(inpt)
        # yt = self.df_conv3d(yt)
        yt = self.conv3d_1(yt)
        ys = self.spt_att(inpt)
        ys = self.df_conv2d(ys).reshape(inpt.shape)
        ys = self.conv3d_2(ys)
        return self.alpha * ys + (1 - self.alpha) * yt


# endregion # <--- STDA ---> #

class STDAResNeXt3D(nn.Module):
    def __init__(self, layers, cardinality=32, STDA_input_shape=[1024, 2, 7, 7], input_channels=3, conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type="B", device="cpu", **kwargs):
        super().__init__()
        self.in_planes = 128
        self.cardinality = cardinality
        self.STDA_input_shape = STDA_input_shape
        self.device = device
        self.conv1 = nn.Conv3d(input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResNeXtBottleneck3D, layers[0], 128,
                                       shortcut_type)
        self.layer2 = self._make_layer(ResNeXtBottleneck3D,
                                       layers[1],
                                       256,
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(ResNeXtBottleneck3D,
                                       layers[2],
                                       512,
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(STDABottleneck,
                                       layers[3],
                                       1024,
                                       shortcut_type,
                                       stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if self.device == "cuda":
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, blocks, planes, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(inplanes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  cardinality=self.cardinality,
                  downsample=downsample,
                  input_shape=self.STDA_input_shape))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.in_planes,
                                planes=planes,
                                stride=1,
                                cardinality=self.cardinality,
                                input_shape=[self.STDA_input_shape[0] * block.expansion,
                                             int(np.ceil(self.STDA_input_shape[1] / stride)),
                                             int(np.ceil(self.STDA_input_shape[2] / stride)),
                                             int(np.ceil(self.STDA_input_shape[3] / stride))]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNeXt3D(nn.Module):
    def __init__(self, layers, cardinality=32, STDA_input_shape=(1024, 2, 7, 7), input_channels=3, conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type="B", device="cpu", **kwargs):
        super().__init__()

        self.in_planes = 128
        self.cardinality = cardinality
        self.STDA_input_shape = STDA_input_shape
        self.device = device
        self.conv1 = nn.Conv3d(input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResNeXtBottleneck3D, layers[0], 128,
                                       shortcut_type)
        self.layer2 = self._make_layer(ResNeXtBottleneck3D,
                                       layers[1],
                                       256,
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(ResNeXtBottleneck3D,
                                       layers[2],
                                       512,
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(ResNeXtBottleneck3D,
                                       layers[3],
                                       1024,
                                       shortcut_type,
                                       stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if self.device == "cuda":
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, blocks, planes, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(inplanes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  cardinality=self.cardinality,
                  downsample=downsample,
                  input_shape=self.STDA_input_shape))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.in_planes,
                                planes=planes,
                                stride=1,
                                cardinality=self.cardinality,
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnext3d50(**kwargs):
    model = ResNeXt3D(layers=[3, 4, 6, 3], **kwargs)
    return model


def resnext3d101(**kwargs):
    model = ResNeXt3D(layers=[3, 4, 23, 3], **kwargs)
    return model


def resnext3d152(**kwargs):
    model = ResNeXt3D(layers=[3, 4, 36, 3], **kwargs)
    return model


def stda_resnext3d50(**kwargs):
    model = STDAResNeXt3D(layers=[3, 4, 6, 3], **kwargs)
    return model


def stda_resnext3d101(**kwargs):
    model = STDAResNeXt3D(layers=[3, 4, 23, 3], **kwargs)
    return model


def stda_resnext3d152(**kwargs):
    model = STDAResNeXt3D(layers=[3, 4, 36, 3], **kwargs)
    return model
