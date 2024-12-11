import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from math import sqrt

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff
        

class Adaptive_Perspective_Transformation(nn.Module):

    def __init__(self):
        super(Adaptive_Perspective_Transformation, self).__init__()
        # self.corner_offsets = corner_offsets.view(-1,4,2)
        self.fc = nn.Sequential(
            nn.Linear(128, 9)  
        )
        self.device = 'cuda'

    def _getPerspectiveTransform(self,src,dst):
        bs, _, _ = src.size()
        A = torch.zeros((bs, 8, 8), dtype=torch.float32, device = self.device)
        A[:, 0::2, 0:3] = torch.cat([src[:, :, 0:1], src[:, :, 1:2], torch.ones((bs, 4, 1), device = self.device)], dim=2)
        A[:, 1::2, 3:6] = torch.cat([src[:, :, 0:1], src[:, :, 1:2], torch.ones((bs, 4, 1), device = self.device)], dim=2)
        A[:, 0::2, 6:8] = -dst[:, :, 0:1] * src
        A[:, 1::2, 6:8] = -dst[:, :, 1:2] * src
        B = dst.reshape(bs, -1, 1)
        #half转float
        A = A.float()
        B = B.float()
        H = torch.linalg.solve(A, B)
        perspective_transform = torch.cat([H, torch.ones((bs, 1, 1),device = self.device)], dim=1).reshape(bs, 3, 3)
        return perspective_transform

    def _warp_perspective(self,img, M, output_shape):
        B, _, _, _ = img.shape
        out_w, out_h = output_shape[:2]
        x1, y1 = torch.meshgrid(torch.arange(out_w, device =img.device ), torch.arange(out_h, device =img.device ))  # shape: (out_h, out_w)
        x1 = x1.t()
        y1 = y1.t()
        grid_out = torch.vstack([x1.ravel().double(), y1.ravel().double(), torch.ones_like(x1.ravel()).double()]).T  
        grid_out = grid_out.repeat(B, 1, 1)
        M = M.to(img.device)
        grid_in = torch.bmm(torch.linalg.inv(M).float(), grid_out.float().transpose(1, 2))
        grid_in = grid_in[:, :2, :] / grid_in[:, 2:, :]
        grid_in = grid_in.view(B, 2, out_h, out_w).permute(0, 2, 3, 1)
        grid_in0 = grid_in[..., 0] / ((out_w - 1) / 2) - 1
        grid_in1 = grid_in[..., 1] / ((out_h - 1) / 2) - 1
        grid_in = torch.stack((grid_in0, grid_in1), dim=-1)
        img_warped = F.grid_sample(img, grid_in, mode='bilinear', align_corners=False)
        return img_warped
    
    def _refine_M(self, M, off_feat):
        feat = self.fc(off_feat).view(M.size(0),3,3)
        M = torch.mul(feat,M)
        M[:, 2, 2] = 1
        return M


    def forward(self, x,corner_offsets):
        N,_,ih,iw = x.size()
        self.device = x.device
        corner_offsets = corner_offsets.view(-1,4,2)
        corner = torch.tensor([[0, 0],
                       [0, ih],
                       [iw, 0],
                       [ih, iw]], device = self.device).unsqueeze(0).repeat(N,1,1)
        corner_trans = corner + corner_offsets.to(self.device)
        M = self._getPerspectiveTransform(corner,corner_trans)
        # M = self._refine_M(M, off_feat)
        img_warped = self._warp_perspective(x, M, [ih,iw])
        return img_warped

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):

        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
    
    
    
def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module