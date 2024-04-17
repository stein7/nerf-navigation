from torch.autograd import Function 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np

import pdb
import numpy as np

import torch.nn.functional as F

import math


class DH(nn.Linear):
    def __init__(self, opt, in_features, out_features, bias, layer, training=False,  
                 a_star=False, bp=False):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.training = training
        self.a_star = a_star
        self.bp = bp
        self.layer = layer
        self.opt = opt

        self.approx = [0.501, 0.553, 0.571, 0.573] 
        self.msb_compen = [0.295, 0.207, 0.211, 0.223]  
        
    def forward(self, x):
        if self.training            : out = Accurate.apply(x, self.weight.T, self)
        elif self.a_star            : out = A_STAR.apply( x, self.weight.T, self)
        elif self.bp                : out = BP.apply( x, self.weight.T, self )
        else                        : out = Accurate.apply(x, self.weight.T, self) 
        return out
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
class Accurate(Function):
    @staticmethod
    def forward(ctx, x, w, L):
        print(f'Default Mode !!!') 
        out = torch.mm(x, w)
        ctx.constant = (x, w, L)
        return out.type(torch.float32)
    @staticmethod
    def backward(ctx, grad_output):
        x, w, L = ctx.constant
        err = torch.mm( grad_output, w.T )
        w_grad = torch.mm( x.T.type(torch.float32), grad_output.type(torch.float32))
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None
    
## L.layer 정보? 위에선 self.layer = 0 이정도만
class A_STAR(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
        print(f'A Star Inference Mode - Layer {L.layer} !!!') 
        if L.layer == 0                         : out = __A_STAR_MSB_COMPEN__( x, w, L )                           # MSB Compen
        elif L.layer > 0 and L.layer < 3        : out = __A_STAR_APPROX__( x, w, L )                            # Approx w/ W Alloc + I/W Switching
        elif L.layer == 3                       : out = __A_STAR_MSB_COMPEN__( x, w, L )                           # MSB Compen
        
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        print(f'A star backward!!')
        x, w, L = ctx.constant

        if L.layer == 3                         : err = torch.mm(grad_output, w.T)                       # Accurate
        elif L.layer<3 and L.layer>0            : err = __A_STAR_APPROX__(grad_output, w.T, L)
        else                                    : err = torch.mm(grad_output, w.T) 

        w_grad = torch.mm(x.T, grad_output)

        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class BP(Function):
    @staticmethod
    def forward( ctx, x, w, L ): 
        print(f'BP Inference Mode !!!') 
        out = __A_STAR_MSB_COMPEN__( x, w, L ) 
        ctx.constant = ( x, w, L )
        return out
    def backward( ctx, grad_output ):
        x, w, L = ctx.constant 
        print(f'BP Inference Mode !!!') 
        err = __A_STAR_MSB_COMPEN__(grad_output, w.T, L)                       # Accurate
        w_grad = torch.mm(x.T, grad_output)
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None


@torch.no_grad()
def __A_STAR_APPROX__(x, w, L):
    #print(f'INIT Compressor-based APPROX Comp Layer {L.layer}')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.approx[L.layer] * L.opt.a_star_approx_scaling
    print(f'dh a star approx scaling factor is { L.opt.a_star_approx_scaling }') 
    return out
@torch.no_grad()
def __A_STAR_MSB_COMPEN__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.msb_compen[L.layer] * L.opt.a_star_MSB_scaling
    print(f'dh a star MSB scaling factor is { L.opt.a_star_MSB_scaling }') 
    return out

# @torch.no_grad()
# def __SWITCHING__(x, w, L):
#     #print(f'INIT Compressor-based MSB Compen Comp')
#     out = torch.mm( x, w )
#     out = out + torch.abs(out).mean() * L.switching[L.layer]
#     return out
# @torch.no_grad()
# def __MSB_COMPEN_SWITCHING__(x, w, L):
#     #print(f'INIT Compressor-based MSB Compen Comp')
#     out = torch.mm( x, w )
#     out = out + x.mean() * L.msb_compen_switching[L.layer]
#     return out
