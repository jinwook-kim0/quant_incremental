import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import numpy as np



def LinQF(fxbits=4, fwbits=4, bxbits=1, bwbits=1, bybits=1, ay = 2):

    class cLinQF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, b, ax, aw, mode):
            x = x.div(ax)
            x_c = x.clamp(max = 1)
            x_q = uniform_quant(x_c, fxbits)
            x_qm = x_q.mul(ax)

            w = weight.div(aw)
            w_c = w.clamp(min=-1, max=1)
            sign = w_c.sign()
            w_abs = w_c.abs()
            w_q = uniform_quant(w_abs, fwbits).mul(sign)
            w_qm = w_q.mul(aw)

            b_t  = b.div(aw * ax).clamp(min=-1, max=1)
            bs = b_t.sign()
            ba = b_t.abs()
            bq = uniform_quant(ba, fwbits).mul(bs)
            bqm = bq.mul(aw*ax)

'''
def Conv2dF(fxbits=4, fwbits=4, bxbits=1, bwbits=1, bybits=1, qf=lambda x: 2 * x.abs().mean()):
    class cConv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, ax, aw, bias=None, mode=True, stride=1, padding=0, dilation=1, groups=1, module=None):
            xq, xd, ra = uniform_quantu(x, ax, fxbits)
            xqm = xq.mul(ra)

            wq, wd, rw = uniform_quants(weight, aw, fwbits)
            wqm = wq.mul(rw)

            ctx.save_for_backward(xq, xd, wq, wd, bias, xqm, wqm)
            ctx.module = module
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.mode = mode
            ctx.fxbits = fxbits
            ctx.fwbits = fwbits
            ctx.bxbits = bxbits
            ctx.bwbits = bwbits
            ctx.ax = ax
            ctx.aw = aw
            ctx.ra = ra
            ctx.rw = rw

            return torch.nn.functional.conv2d(input=xqm, weight=wqm, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        @staticmethod
        def backward(ctx, grad_outputr):
            x_q, xd, w_q, wd, bias, x_qm, w_qm = ctx.saved_tensors
            module = ctx.module
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups
            mode = ctx.mode
            ra = ctx.ra
            rw = ctx.rw
            ax = ctx.ax
            aw = ctx.aw

            grad_input = grad_weight = grad_bias = None
            grad_output = grad_outputr.clone().detach()

            if mode == 3:
                gm = qf(grad_output)
                if gm != 0:
                    gmq, gmd, grm = uniform_quants(grad_output, bybits)
                    go = gmq.mul(grm)
                else:
                    go = grad_output

                x_qmt = tshift(x_q, fxbits, bxbits).mul(ra)
                w_qmt = tshift(w_q.abs(), fwbits, bwbits).mul(w_q.sign()).mul(rw)
            else:
                go = grad_output
                x_qmt = x_qm
                w_qmt = w_qm


            grad_input = torch.nn.grad.conv2d_input(x_qmt.shape, w_qmt, go, stride, padding, dilation, groups)
            grad_weight = torch.nn.grad.conv2d_weight(x_qmt, w_qmt.shape, go, stride, padding, dilation, groups)
            if bias is not None and ctx.needs_input_grad[4]:
                grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

            i = (xd > 1.).float()
            grad_ax = (grad_input * (i + (x_q - xd) * (1-i))).sum()
            grad_input = grad_input * (1-i)

            i = (wd.abs()>1.).float()
            sign = wd.sign()
            grad_aw = (grad_weight * (sign * i + (w_q - wd)*(1-i))).sum()

            if not ctx.needs_input_grad[0]:
                grad_input = None
            if not ctx.needs_input_grad[1]:
                grad_weight = None

            if mode == 1:
                grad_ax = None
            elif mode == 2:
                grad_ax = None
                grad_aw = (grad_weight * (sign * i + (w_q - wd)*(1-i))).sum()

            elif mode == 3:
                gm = grad_output.abs().mean()
                ag = gm * ay
                if ag != 0:
                    gmq = uniform_quant(grad_output.div(ag).abs(), bybits)
                    gmq = gmq.mul(grad_output.sign()).mul(ag)
                else:
                    gmq = grad_output

                x_qmt = right_shift(x_q, fxbits, bxbits).mul(ax)
                w_qmt = right_shift(w_q.abs(), fwbits, bwbits).mul(w_q.sign()).mul(aw)

                grad_input = torch.nn.grad.conv2d_input(x_qm.shape, w_qmt, gmq, stride, padding, dilation, groups)
                grad_weight = torch.nn.grad.conv2d_weight(x_qmt, weight.shape, gmq, stride, padding, dilation, groups)
                #print(grad_input)
                #print(grad_weight)
                if bias is not None and ctx.needs_input_grad[4]:
                    grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

                i = (x > 1.).float()
                grad_ax = None
                grad_input = grad_input * (1-i)

                grad_aw = None # (grad_weight * (sign * i + (w_q - weight)*(1-i))).sum()
                if not ctx.needs_input_grad[0]:
                    grad_input = None
                if not ctx.needs_input_grad[1]:
                    grad_weight = None

            return grad_input, grad_weight, grad_ax, grad_aw, grad_bias, None, None, None, None, None, None

    return cConv().apply
'''


def weight_quantization(b):
    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)  
        #print('uniform quant bit: ', b)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()     # output matrix is a form of [True, False, True, ...]
            sign = input.sign()              # output matrix is a form of [+1, -1, -1, +1, ...]
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            # above line, if i = True,  and sign = +1, "grad_alpha = grad_output * 1"
            #             if i = False, and sign = -1, "grad_alpha = grad_output * (input_q-input)"
            return grad_input, grad_alpha

    return _pq().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        mean = weight.data.mean()
        std = weight.data.std()
        weight = weight.add(-mean).div(std)      # weights normalization
        weight_q = self.weight_q(weight, self.wgt_alpha)
        
        return weight_q


class weight_quantize_fnn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        weight_q = self.weight_q(weight, self.wgt_alpha)
        
        return weight_q


def act_quantization(b):

    def uniform_quant(x, b=4):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)  # Mingu edited for Alexnet
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = 4
        self.weight_quant = weight_quantize_fn(w_bit=self.bit)
        self.act_alq = act_quantization(self.bit)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))
        self.weight_q  = torch.nn.Parameter(torch.zeros([out_channels, in_channels, kernel_size, kernel_size]))
        
    def forward(self, x):
        weight_q = self.weight_quant(self.weight)       
        self.weight_q = torch.nn.Parameter(weight_q)  # Store weight_q during the training
        x = self.act_alq(x, self.act_alpha)
        y = F.conv2d(x, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y
    
    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))

'''
class LinearQ(nn.Module):
    def __init__(self, inf, outf, bias=True, device=None, dtype=None, bx=4, bw=4, bb=4):
        super(self).__init__(inf, outf, bias, device, dtype)
        self.layer_type = 'q-Linear'
        self.bx = bx
        self.bw = bw
        self.bb = bb
        self.act_alq = act_quantization(self.bx)
        self.weight_quant = weight_quantize_fn(w_bit=self.bw)
        self.act_alpha = torch.nn.Parmaeter(torch.tensor(8.0))

    def forward(self, x):
        x = self.act_alq(x, self.act_alpha)
        w = self.weight_quant(self.weight, k)

'''


