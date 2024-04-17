import torch
import numpy as np
import pdb

from lib.w_alloc import *

def dec2bin(x, bits):
    mask = 2 ** torch.arange( bits -1 , -1, -1 ).to( x.device, x.dtype )
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()

def bin2dec(b, bits):
    mask = 2 ** torch.arange( bits -1 , -1, -1).to(b.device, b.dtype)
    return torch.sum(mask*b, -1)

def Dec2Bin(x, bits):
    nz_x = x[x!=0]
    nz_x_min = torch.abs(nz_x).min()
    x_int = (x/nz_x_min).to(torch.int)
    
    binary = dec2bin( x_int, bits=bits )

    return binary, nz_x_min

def sparsity(x, w, x_bw, w_bw):

    xbin, x_bias = Dec2Bin( x, x_bw )
    wbin, w_bias = Dec2Bin( w, w_bw )
    
    # weight alloc
    alloc_switching(wbin, xbin)

