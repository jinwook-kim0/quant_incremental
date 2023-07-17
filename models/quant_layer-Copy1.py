import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import numpy as np

def fout_quant(x, b, alpha):

    sign = x.sign()
    input_abs = x.abs()
    
    input_abs = input_abs / alpha                          # weights are first divided by alpha
    input_abs = input_abs.clamp(max=1)       # then clipped to [-1,1]
    xdiv = input_abs.mul(2 ** (b-1) - 1)
    xhard = xdiv.floor().div(2 ** (b-1) - 1).mul(sign).mul(alpha)
     
    return xhard


def psum_quant(x, b, alpha):
        
    sign = x.sign()
    input_abs = x.abs()
        
    input_abs = input_abs / alpha                          # weights are first divided by alpha
    input_abs = input_abs.clamp(max=1)       # then clipped to [-1,1]    
    xdiv = input_abs.mul(2 ** (b-1) - 1)
    xhard = xdiv.floor().div(2 ** (b-1) - 1).mul(sign).mul(alpha)
     
    return xhard   


def conv2d_emul(x, bit, act_alpha, wgt_alpha, weight_q, bias, stride_tuple, padding_tuple, dilation, groups):

    padding = padding_tuple[0]
    stride = stride_tuple[0]
    w_bin = weight_q/(   wgt_alpha/(pow(2,bit-1)-1)    )  # -1 is due to pos / neg values
    x_bin = x/(act_alpha/(pow(2,bit)-1))   # size: in_ch, ni, nj

    ##### lossy compression ####
    x_bin[x_bin<1] = 0
    ############################

    nig = range(x_bin.size(2))
    njg = range(x_bin.size(3))

    ic_tileg = range(int(w_bin.size(1)/64))
    oc_tileg = range(int(w_bin.size(0)/64))

    icg = range(int(w_bin.size(1)))  # input channel 
    ocg = range(int(w_bin.size(0)))  # output channel

    kig = range(w_bin.size(2))
    kjg = range(w_bin.size(3))

    x_pad = torch.zeros(x_bin.size(0), x_bin.size(1), len(nig)+padding*2, len(njg)+padding*2)
    x_pad[:, :, padding:padding+len(nig), padding:padding+len(njg)] = x_bin
    x_pad = x_pad.cuda()

    psum_size_x = int((len(nig)+2*padding - 1)/stride + 1) 
    psum_size_y = int((len(njg)+2*padding - 1)/stride + 1) 
    psum = torch.zeros(x_bin.size(0), len(ic_tileg), len(oc_tileg), 64, psum_size_x, psum_size_y, len(kig), len(kjg)) # output channel size, ij size, w1, w2 size
    psum_nopad = torch.zeros(x_bin.size(0), len(ic_tileg), len(oc_tileg), 64, len(nig), len(njg), len(kig), len(kjg)) # output channel size, ij size, w1, w2 size
    
    psum = psum.cuda()
    psum_nopad = psum_nopad.cuda()

    for ki in kig:
        for kj in kjg:
            for ic_tile in ic_tileg:       # Tiling into 64X64 array
                for oc_tile in oc_tileg:   # Tiling into 64X64 array        
                    for ni in nig:         # time domain, sequentially given input
                        for nj in njg:     # time domain, sequentially given input
                            m = nn.Linear(64, 64, bias=False)
                            m.weight = torch.nn.Parameter(w_bin[oc_tile*64:(oc_tile+1)*64, ic_tile*64:(ic_tile+1)*64, ki, kj])
                            psum_temp = m(x_bin[:,ic_tile*64:(ic_tile+1)*64,ni,nj]).cuda()
                            psum_nopad[:, ic_tile, oc_tile, :, ni, nj, ki, kj] = psum_temp
        
        
    #psum_nopad = psum_quant(psum_nopad, 3, 2**11, 0)                                                        
    psum[:,:,:,:,padding:padding+len(nig),padding:padding+len(njg),:,:] = psum_nopad 
    
    fout_size_x = int((len(nig)+2*padding -(w_bin.size(2) - 1) - 1)/stride + 1) 
    fout_size_y = int((len(njg)+2*padding -(w_bin.size(3) - 1) - 1)/stride + 1) 
    
    
    fout = torch.zeros(x_bin.size(0), len(ocg), fout_size_x, fout_size_y)
    fout = fout.cuda()
   
   
    ## SFP accumulation ###
    for ni in range(0, fout_size_x):
        for nj in range(0, fout_size_y):
            for ki in kig:
                for kj in kjg:
                    for ic_tile in ic_tileg:    
                        for oc_tile in oc_tileg:   
                            fout[:,oc_tile*64:(oc_tile+1)*64, ni, nj] = fout[:,oc_tile*64:(oc_tile+1)*64, ni, nj] + \
                            psum[:,ic_tile, oc_tile, :, ni+ki, nj+kj, ki, kj]

    #fout = fout_quant(fout, 7, 2**15, 0) 

    fout = fout * (act_alpha/(pow(2,bit)-1)) * (wgt_alpha/(pow(2,bit-1)-1)) 
    
    return fout




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


def act_quantization(b):

    def uniform_quant(x, b=3):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, emul=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = 4
        self.weight_quant = weight_quantize_fn(w_bit=self.bit)
        self.act_alq = act_quantization(self.bit)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))
        
    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        self.register_parameter('weight_q', Parameter(weight_q))  # Mingu added
        x = self.act_alq(x, self.act_alpha)
        return F.conv2d(x, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)
