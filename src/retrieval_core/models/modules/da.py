import torch
from torch import nn

"""
Attention module as implemented in "Dual Attention Network for Scene 
Segmentation" https://arxiv.org/abs/1809.02983
"""

class ActivatedBatchNorm(nn.Module):
    def __init__(self, num_features, activation='relu', **kwargs):
        """
        Pre-activates tensor with activation function before applying batch norm.
        See following link for details. Leads to better performance.
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md

        :param num_features: number of incoming feature maps
        :param activation: activation type
        :param kwargs: key word arguments pertaining to BatchNorm
        """
        super().__init__()

        activation_map = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
        }
        if activation not in activation_map:
            self.act = None
        else:
            self.act = activation_map[activation](inplace=True)
        self.bn = nn.BatchNorm2d(num_features, **kwargs)

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        x = self.bn(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1):
        """
        Conv 3x3
        :param in_dim: input channels
        :param out_dim: output_channels
        :param kernel_size:
        :param padding:
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvPreAct(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1):
        """
        Conv 3x3 -> activation -> BatchNorm
        :param in_dim: input channels
        :param out_dim: output_channels
        :param kernel_size:
        :param padding:
        """
        super().__init__()
        self.conv = Conv3x3(in_dim, out_dim, kernel_size, padding)
        self.act = ActivatedBatchNorm(out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


# Both PAModule & CAModule are taken from Dual attention network as per,
# https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
# See https://arxiv.org/pdf/1809.02983.pdf

class PAModule(nn.Module):
    def __init__(self, in_dim):
        """
        input feature maps( B X C X H X W)
        Position attention module
        Here, the generated attention map is based on the shape of the spatial
        dimensions B x (H x W) x (H x W)
        """
        super(PAModule, self).__init__()
        self.in_dim = in_dim

        self.query_conv = Conv1x1(self.in_dim, self.in_dim // 8)
        self.key_conv = Conv1x1(self.in_dim, self.in_dim // 8)
        self.value_conv = Conv1x1(self.in_dim, self.in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, channels, height, width = x.size()

        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, channels, height, width)

        out = self.gamma*out + x
        return out


class CAModule(nn.Module):
    """
    input feature maps( B X C X H X W)
    Channel attention module
    Here, the generated attention map is based on the shape of the channel
    dimensions B x (C x C)
    """
    def __init__(self, in_dim):
        super(CAModule, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DAModule(nn.Module):
    def __init__(self, in_dim):
        """
        Dual attention module from https://arxiv.org/pdf/1809.02983.pdf
        Features from CAM and PAM are summed
        :param in_dim:input dimensions
        """
        super(DAModule, self).__init__()

        inter_dim = in_dim // 4
        self.conv_pam1 = ConvPreAct(in_dim, inter_dim)
        self.pam = PAModule(inter_dim)
        self.conv_pam2 = ConvPreAct(inter_dim, inter_dim)

        self.conv_cam1 = ConvPreAct(in_dim, inter_dim)
        self.cam = CAModule(inter_dim)
        self.conv_cam2 = ConvPreAct(inter_dim, inter_dim)

        self.conv = ConvPreAct(inter_dim, in_dim)
        self.out_dim = in_dim

    def forward(self, x):
        p = self.conv_pam1(x)
        p = self.pam(p)
        p = self.conv_pam2(p)

        c = self.conv_cam1(x)
        c = self.cam(c)
        c = self.conv_cam2(c)

        feat = p + c
        feat = self.conv(feat)
        return feat

