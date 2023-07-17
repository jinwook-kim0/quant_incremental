
import torch
import torch.nn as nn
import torch.nn.grad as G
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def getbw(x):
    try:
        xim = x.abs().int().max()
        return int(np.log2(xim.cpu()).ceil())
    except:
        print(x.abs().max())

def bitshrink(x, b):
    xi = x >> b
    return xi << b

def uniform_quantu(x, b, f=lambda x: x.round()):
    xc = x.clamp(max=1)
    xm = xc.mul(2 ** b - 1)
    xq = f(xm).div(2 ** b - 1)
    return xq

def uniform_quants(x, b, f=lambda x: x.round()):
    xc = x.clamp(min=-1, max=1)
    xs = xc.sign()
    xa = xc.abs()
    if b > 0:
        return f(xa.mul(2 ** b - 1)).mul(xs).div(2 ** b - 1)
    else: 
        return xs

def get_intm(x, b):
    return (x.mul(2 ** b - 1)).int()

def tshift(x, b0, b):
    xi = x.mul(2** b0 - 1).int()
    xs = (xi >> b) if b > 0 else (xi << b)
    return torch.tensor((x.div(2**b0-1)), dtype=x.dtype)

def get_mask(x, b):
    return torch.ones(x.shape, dtype=torch.int32) * b

def masking(x, b, m):
    xi = x.mul(2**b - 1).int()
    xm = x & m
    return torch.tensor(x.div(2**b - 1), dtype=x.dtype)

def Conv2dQ(xb=4, wb=4):
    class qConv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, ax, aw, bias=None, stride=1, padding=0, dilation=1, groups=1):
            xd = x.div(ax)
            xdq = uniform_quantu(xd, xb)
            xq = xdq.mul(ax)

            wd = w.div(aw)
            wdq = uniform_quants(wd, wb)
            wq = wd.mul(aw)

            ctx.save_for_backward(x, xd, xdq, xq, w, wd, wdq, wq, bias)
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.ax = ax
            ctx.aw = aw
            return F.conv2d(xq, wq, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        @staticmethod
        def backward(ctx, gy):
            x, xd, xdq, xq, w, wd, wdq, wq, bias = ctx.saved_tensors
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups
            ax = ctx.ax
            aw = ctx.aw

            gi = gw = gb = None
            gax = gaw = None

            gi = G.conv2d_input(xq.shape, wq, gy, stride, padding, dilation, groups)
            gw = G.conv2d_weight(xq, wq.shape, gy, stride, padding, dilation, groups)

            xdc = (xd > 1.0).float()

            if ctx.needs_input_grad[2]:
                gax = (gi * (xdc + (xdq - xd)*(1-xdc))).sum()
            gi = gi * (1 - xdc)

            if ctx.needs_input_grad[3]:
                wdc = (wd.abs()>1.0).float()
                sign = wd.sign()
                gaw = (gw *  (sign * wdc + (wdq - wd) * (1 - wdc))).sum()


            if not ctx.needs_input_grad[0]:
                gi = None
            if not ctx.needs_input_grad[1]:
                gw = None

            if bias is not None and ctx.needs_input_grad[4]:
                gb = gy.sum((0, 2, 3)).squeeze(0)

            return gi, gw, gax, gaw, gb, None, None, None, None
    return qConv().apply

def Conv2dQB(fxbits=4, fwbits=4, bxbits=1, bwbits=1, bybits=1, bibits=1, qf=lambda x: x.abs().mean() * 2):
    class qConv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, ax, aw, bias=None, stride=1, padding=0, dilation=1, groups=1):
            xd = x.div(ax)
            xdq = uniform_quantu(xd, fxbits)
            xq = xdq.mul(ax)

            wd = weight.div(aw)
            wdq = uniform_quants(wd, fwbits)
            wq = wd.mul(aw)

            ctx.save_for_backward(x, xd, xdq, xq, weight, wd, wdq, wq, bias)
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.ax = ax
            ctx.aw = aw

            return F.conv2d(xq, wq, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        @staticmethod
        def backward(ctx, gy):
            x, xd, xdq, xq, w, wd, wdq, wq, bias = ctx.saved_tensors
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups
            ax = ctx.ax
            aw = ctx.aw

            gi = gw = gb = None
            gax = gaw = None

            ay = qf(gy)
            gydq = gy

            if ay != 0:
                gyd = gy.clone().detach().div(ay)
                gydq = uniform_quants(gyd, bybits).mul(ay) # mul(2 ** bybits - 1)

            xqb = bitshrink(get_intm(xdq, fxbits), fxbits - bxbits).mul(ax / (2 ** fxbits - 1)) # + 0.0
            wqb = bitshrink(get_intm(wdq.abs(), fwbits), fwbits - bwbits).mul(wdq.sign()).mul(aw / (2 ** fwbits-1)) # + 0.0

            gi = G.conv2d_input(xq.shape, wqb, gydq, stride, padding, dilation, groups)
            gw = G.conv2d_weight(xqb, wq.shape, gy, stride, padding, dilation, groups)

            xdc = (xd > 1.0).float()
            if ctx.needs_input_grad[2]:
                gax = (gi * (xdc + (xdq - xd)*(1-xdc))).sum()

            gi = (gi * (1 - xdc))
            if bibits != 0:
                gibw = getbw(gi) # 사실 x, w bitwidth에 따라서 계산 가능
                gi = bitshrink(gi.int().abs(), gibw - bibits).mul(gi.sign())

            if ctx.needs_input_grad[3]:
                wdc = (wd.abs()>1.0).float()
                sign = wd.sign()
                gaw = (gw *  (sign * wdc + (wdq - wd) * (1 - wdc))).sum()

            if not ctx.needs_input_grad[0]:
                gi = None
            if not ctx.needs_input_grad[1]:
                gw = None

            if bias is not None and ctx.needs_input_grad[4]:
                gb = gy.sum((0, 2, 3)).squeeze(0)

            return gi, gw, gax, gaw, gb, None, None, None, None
    return qConv().apply



class QuantConv2df(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, func=Conv2dQ(4, 3), bits=3, acbits=7, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2df, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))
        self.wgt_alpha = torch.nn.Parameter(torch.tensor(3.0))
        self._freezed = False
        self.bits = bits
        self.acbits = acbits
        self.F = func


    def forward(self, x):
        weight = self.weight.add(-self.weight.data.mean()).div(self.weight.data.std())
        y = self.F(x, weight, self.act_alpha, self.wgt_alpha, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


    def update_func(self, func):
        self.F = func


