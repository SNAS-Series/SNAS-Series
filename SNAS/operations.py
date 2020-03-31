import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'hard_none': lambda C, stride, affine: HardZero(stride),
    'avg_pool_3x3': lambda C, stride, affine: AvgPool(C, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: MaxPool(C, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class AvgPool(nn.Module):
    def __init__(self, C, stride, padding, count_include_pad=False, affine=True):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.C = C
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=count_include_pad),
            nn.BatchNorm2d(C, affine=affine)
        )
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = 2 * (y ** 2) * self.C
            else:
                y = original_h // 2
                self.mac = 5 * (y ** 2) * self.C
            self.flops = 3 * 3 * (y ** 2) * self.C

        return self.op(x)


class MaxPool(nn.Module):
    def __init__(self, C, stride, padding, affine=True):
        super(MaxPool, self).__init__()
        self.C = C
        self.stride = stride
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=padding),
            nn.BatchNorm2d(C, affine=affine)
        )
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = 2 * (y ** 2) * self.C
            else:
                y = original_h // 2
                self.mac = 5 * (y ** 2) * self.C
            self.flops = 3 * 3 * (y ** 2) * self.C

        return self.op(x)


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        self.size = C_in * C_out * kernel_size * kernel_size
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = (y ** 2) * (self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = (y ** 2) * (4 * self.C_in + self.C_out) + self.size
            self.flops = self.size * (y ** 2)

        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        self.size = C_in * kernel_size * kernel_size + C_out * C_in
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = (y ** 2) * (3 * self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = (y ** 2) * (6 * self.C_in + self.C_out) + self.size
            self.flops = self.size * (y ** 2)

        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        self.size = 2 * C_in * (kernel_size ** 2) + (C_in + C_out) * C_in
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.flops == 0:
            original_h = x.size()[2]
            if self.stride == 1:
                y = original_h
                self.mac = (y ** 2) * (7 * self.C_in + self.C_out) + self.size
            else:
                y = original_h // 2
                self.mac = (y ** 2) * (10 * self.C_in + self.C_out) + self.size
            self.flops = self.size * (y ** 2)

        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.mac == 0:
            y = x.size()[2]
            c = x.size()[1]
            self.mac = 2 * c * (y ** 2)
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class HardZero(nn.Module):
    def __init__(self, stride):
        super(HardZero, self).__init__()
        self.stride = stride
        self.size = 0
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        return 0


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.size = 2 * C_in * (C_out // 2)
        self.flops = 0
        self.mac = 0

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        if self.flops == 0:
            original_h = x.size()[2]
            y = original_h // 2
            self.flops = self.size * (y ** 2)
            self.mac = 2 * (y ** 2) * (4 * self.C_in + self.C_out // 2) + self.size
        return out
