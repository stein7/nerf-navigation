from torch.autograd import Function 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np

import pdb
import numpy as np

from lib.precision import *
from lib.writer import *
import torch.nn.functional as F

import math
from lib.bit import *

class CFC(nn.Linear):
    def __init__(self, in_features, out_features, bias, training=False, inf_accurate=False, bp_accurate=False, eval_ich_range=False, inf_approx=False, inf_msb=False, inf_switching=False, inf_w_alloc=False, a_star=False, tuning=False):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.training = training
        self.eval_ich_range = eval_ich_range
        self.inf_approx = inf_approx
        self.inf_accurate = inf_accurate
        self.inf_msb = inf_msb
        self.inf_switching = inf_switching
        self.inf_w_alloc = inf_w_alloc
        self.a_star = a_star
        self.tuning = tuning
        self.ich_flag = False
        self.FF_quant = False
        self.BP_quant = False
        self.layer = 0

        self.approx = [0.681, 0.713, 0.751, 0.773]                          # Approx
        self.w_alloc = [0.594, 0.624, 0.689, 0.707]                         # Approx w/ Weight Alloc
        self.switching = [0.49, 0.531, 0.578, 0.603]                         # Approx w/ Weight Alloc + I/W Switching
        self.msb_compen = [0.295, 0.207, 0.211, 0.223]                      # Carry Compensation
        self.msb_compen_w_alloc = [0.175, 0.155, 0.153, 0.153]              # Carry w/ Weight Alloc
        #self.msb_compen_switching = [0.0555, 0.0592, 0.0591, 0.0617]        # Carry w/ Weight Alloc + I/W Switching
        #self.approx = [0.002689313, 0.007436308, 0.03508, 0.389]
        #self.weight_shuffle = [0.001648244, 0.004663158, 0.021886315, 0.323520482]
        #self.switching = [0.001068843, 0.004613975, 0.016299255, 0.29205894]
        #self.msb_compen = [0.008, 0.0022, 0.0121, 0.2175]

        self.inlier = False
        self.bit_sparsity = False
        #self.msb_compen_switching = [0.0555, 0.0592, 0.0591, 0.0617]        # Carry w/ Weight Alloc + I/W Switching
        #self.msb_compen_switching = [0.0755, 0.0892, 0.0991, 0.1017]        # Carry w/ Weight Alloc + I/W Switching
        #self.msb_compen_switching = [0.175, 0.295, 0.413, 0.713]        # Carry w/ Weight Alloc + I/W Switching

        #self.msb_compen_switching = [0.775, 0.895, 0.913, 1.253]        # Carry w/ Weight Alloc + I/W Switching
        #self.msb_compen_switching = [0.375, 0.495, 0.613, 0.753]        # Carry w/ Weight Alloc
        self.msb_compen_switching = [0.095, 0.215, 0.333, 0.473]        # Carry


    def forward( self, x ):
        

        if self.training                        : out = Accurate.apply( x, self.weight.T, self )
        elif self.inf_accurate                  : out = Accurate.apply( x, self.weight.T, self )
        elif self.inf_approx                    : out = APPROX.apply( x, self.weight.T , self)
        elif self.inf_msb                       : out = MSB_COMPEN.apply( x, self.weight.T, self )
        elif self.inf_switching                 : out = SWITCHING.apply( x, self.weight.T, self )
        elif self.inf_w_alloc                   : out = W_ALLOC.apply( x, self.weight.T, self )
        elif self.a_star                        : out = A_STAR.apply( x, self.weight.T, self )
        elif self.tuning                        : out = TUNING.apply( x, self.weight.T, self )
        else                                    : out = Accurate.apply( x, self.weight.T, self )
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Accurate(Function):
    @staticmethod
    def forward( ctx, x, w , L ):

        if L.FF_quant:
            print(f'FF {L.layer}th layer quantization')
            infer_bw = 7
            int_bw = int(math.log2(torch.abs(x).max()))
            x = custom_precision( x, int_bw, infer_bw-int_bw-1, 'fxp' )

            #x_nonz = x[x!=0]
            #hist = torch.histc(x_nonz.reshape(-1).cuda(), bins=100) 

            #maxi = hist.max()
            #maxi_bmap = hist == maxi
            #maxi_idx = torch.nonzero(maxi_bmap)
            #
            #if maxi_idx < hist.shape[0]-1:
            #    for bit in range (1, 4):
            #        b = 2**(bit-1) -1

            #        indexing = hist[maxi_idx : maxi_idx+2*b+1]
            #        print(f'{bit}: {indexing.sum()/hist.sum()}, offloading ich: {128*(1- indexing.sum()/hist.sum() )}')

            #        print(f'===================================================================================================================')


            #bins = torch.linspace(x_nonz.min(), x_nonz.max(), 100)
            #hist = torch.cat( [bins.reshape(1,-1), hist.reshape(1,-1)], dim=0 )
            #np.savetxt('./result/inference_input_l'+str(L.layer)+'.csv', hist.detach().cpu().numpy(), delimiter=',')

            #L.writer.UpdateData(data=x.reshape(-1).detach().cpu().numpy())
            #L.writer.PngHistPlot( file_path='./result/inference_result_l'+str(L.layer)+'.png', title=f'Layer {L.layer} Output Distribution', xlabel=f'Output Value', ylabel='Num', bins=20 )


        if L.ich_flag: np.savetxt('./result/inference_result_l'+str(L.layer)+'.csv', x.detach().cpu().numpy(), delimiter=',')

        out = torch.mm( x, w )
        ctx.constant = (x, w, L)

        
        return out.type( torch.float32 )


    @staticmethod
    def backward( ctx, grad_output ):
        x, w, L = ctx.constant
        #print(f'grad_output:{grad_output.shape}')
        #print(f'weight: {w.shape}')


        if L.BP_quant:
            print(f'BP {L.layer}th layer quantization')
            bp_bw = 8
            int_bw = int(math.log2(torch.abs(grad_output).max()))
            grad_output = custom_precision( grad_output, int_bw, bp_bw-int_bw-1, 'fxp' )

            #grad_nonz = grad_output[grad_output!=0]
            #hist = torch.histc(grad_nonz.reshape(-1).cuda(), bins=100) 

            #maxi = hist.max()
            #maxi_bmap = hist == maxi
            #maxi_idx = torch.nonzero(maxi_bmap)
            #
            #if maxi_idx < hist.shape[0]-1 and maxi_idx > 0:
            #    for bit in range (3, 8):
            #        b = 2**(bit-1) -1

            #        indexing = hist[maxi_idx-b : maxi_idx+b+1]
            #        print(f'{bit}: {indexing.shape} {indexing.sum()/hist.sum()}, offloading ich: {128*(1- indexing.sum()/hist.sum() )}')


            #bins = torch.linspace(grad_output.min(),grad_output.max(), 100)
            #hist = torch.cat( [bins.reshape(1,-1), hist.reshape(1,-1)], dim=0 )
            #np.savetxt('./result/bp_error_l'+str(L.layer)+'.csv', hist.detach().cpu().numpy(), delimiter=',')
            #L.writer.UpdateData(data=grad_output.reshape(-1).detach().cpu().numpy())
            #L.writer.PngHistPlot( file_path='./result/bp_result_l'+str(L.layer)+'.png', title=f'Layer {L.layer} Err Distribution', xlabel=f'Err Value', ylabel='Num', bins=20 )


        if L.ich_flag : np.savetxt('./result/backward_result_l'+str(L.layer)+'.csv', grad_output.detach().cpu().numpy(), delimiter=',')

        err = torch.mm( grad_output, w.T )


        w_grad = torch.mm( x.T.type(torch.float32), grad_output.type(torch.float32) )
        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class TUNING(Function):
    @staticmethod
    def forward( ctx, x, w, L ): 
        print(f'MSB Compensation w/ Weight Alloc & I/W Switching Inference - Layer {L.layer} !!!') 

        if L.FF_quant: 
            infer_bw = 8
            int_bw = int(math.log2(torch.abs(x).max()))
            x = custom_precision( x.clone().detach(), int_bw, infer_bw-int_bw-1, 'fxp' ).clone().detach().requires_grad_(True)

        if L.inlier:
            x_nonz = x[x!=0]
            hist = torch.histc(x_nonz.reshape(-1).cuda(), bins=100) 

            maxi = hist.max()
            maxi_bmap = hist == maxi
            maxi_idx = torch.nonzero(maxi_bmap)
            
            if maxi_idx < hist.shape[0]-1:
                for bit in range (1, 6+1):
                    b = 2**(bit-1)
                    
                    start = maxi_idx-b
                    end = maxi_idx+b

                    if maxi_idx-b < 0:
                        #start = maxi_idx
                        #end = maxi_idx + 2*b
                        start = 0
                        end = 2*b

                    if maxi_idx+b > hist.shape[0]: 
                        end = hist.shape[0]
                    
                    print(f'start: {start} ~ end: {end}')

                    indexing = hist[start : end]
                    print(f'{bit} bit batch ratio: {indexing.sum()/hist.sum()}, offloading ch num: {128*(1- indexing.sum()/hist.sum() )}')
                    print(f'===================================================================================================================')


        out = __MSB_COMPEN_SWITCHING__( x, w, L ) 
        ctx.constant = ( x, w, L )
        return out
    def backward( ctx, grad_output ):
        x, w, L = ctx.constant 
        print(f'MSB Compensation w/ Weight Alloc & I/W Switching Backward  - Layer {L.layer} !!!') 

        if L.FF_quant: 
            bp_bw = 8
            int_bw = int(math.log2(torch.abs(grad_output).max()))
            grad_output = custom_precision( grad_output.clone().detach(), int_bw, bp_bw-int_bw-1, 'fxp' ).clone().detach().requires_grad_(True)

        if L.inlier:
            grad_nonz = grad_output[grad_output!=0]
            hist = torch.histc(grad_nonz.reshape(-1).cuda(), bins=100) 

            maxi = hist.max()
            maxi_bmap = hist == maxi
            maxi_idx = torch.nonzero(maxi_bmap)

            if maxi_idx.shape[0] != 1:
                maxi_idx = maxi_idx[0]

            for bit in range (1, 6+1):
                b = 2**(bit-1)
                
                start = maxi_idx-b
                end = maxi_idx+b

                if maxi_idx-b < 0:
                    start = 0
                    end = 2*b
                if maxi_idx == hist.shape[0]-1:
                    start = hist.shape[0]-2*b
                    end = hist.shape[0]

                if maxi_idx+b > hist.shape[0]: 
                    end = hist.shape[0]
                
                print(f'start: {start} ~ end: {end}')

                indexing = hist[start : end]
                print(f'{bit} bit batch ratio: {indexing.sum()/hist.sum()}, offloading ch num: {128*(1- indexing.sum()/hist.sum() )}')
                print(f'===================================================================================================================')
            #if maxi_idx < hist.shape[0]-1:
            #    for bit in range (1, 6+1):
            #        b = 2**(bit-1)
            #        
            #        start = maxi_idx-b
            #        end = maxi_idx+b

            #        if maxi_idx-b < 0:
            #            #start = maxi_idx
            #            #end = maxi_idx + 2*b
            #            start = 0
            #            end = 2*b

            #        if maxi_idx+b > hist.shape[0]: 
            #            end = hist.shape[0]
            #        
            #        print(f'start: {start} ~ end: {end}')

            #        indexing = hist[start : end]
            #        print(f'{bit} bit batch ratio: {indexing.sum()/hist.sum()}, offloading ch num: {128*(1- indexing.sum()/hist.sum() )}')
            #        print(f'===================================================================================================================')

        err = __MSB_COMPEN_SWITCHING__(grad_output, w.T, L)                       # Accurate

        w_grad = torch.mm(x.T, grad_output)

        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class A_STAR(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
        print(f'A Star Inference Mode - Layer {L.layer} !!!') 

        if L.FF_quant: 
            infer_bw = 7
            int_bw = int(math.log2(torch.abs(x).max()))
            x = custom_precision( x.clone().detach(), int_bw, infer_bw-int_bw-1, 'fxp' ).clone().detach().requires_grad_(True)
            
            w_infer_bw = 4
            w_bw = int(math.log2(torch.abs(w).max()))
            w_quant = custom_precision( w.clone().detach(), w_bw, w_infer_bw-w_bw-1, 'fxp' ).clone().detach().requires_grad_(True)

            if L.bit_sparsity:
                # Check Bit Sparsity
                # Check original bit sparsity --> w alloc --> switching
                sparsity( x=x, w=w_quant, x_bw=infer_bw, w_bw=w_infer_bw )

        if L.inlier:
            x_nonz = x[x!=0]
            hist = torch.histc(x_nonz.reshape(-1).cuda(), bins=100) 

            maxi = hist.max()
            maxi_bmap = hist == maxi
            maxi_idx = torch.nonzero(maxi_bmap)
            
            if maxi_idx < hist.shape[0]-1:
                for bit in range (1, 5+1):
                    b = 2**(bit-1)
                    
                    start = maxi_idx-b
                    end = maxi_idx+b

                    if maxi_idx-b < 0:
                        #start = maxi_idx
                        #end = maxi_idx + 2*b
                        start = 0
                        end = 2*b

                    if maxi_idx+b > hist.shape[0]: 
                        end = hist.shape[0]
                    
                    print(f'start: {start} ~ end: {end}')

                    indexing = hist[start : end]
                    print(f'{bit} bit batch ratio: {indexing.sum()/hist.sum()}, offloading ch num: {128*(1- indexing.sum()/hist.sum() )}')
                    print(f'===================================================================================================================')

        #if L.layer == 0                         : out = __SWITCHING__( x, w, L )                           # MSB Compen
        #elif L.layer > 0 and L.layer < 3        : out = __SWITCHING__( x, w, L )                            # Approx w/ W Alloc + I/W Switching
        #elif L.layer == 3                       : out = __SWITCHING__( x, w, L )                           # MSB Compen

        #if L.layer == 0                         : out = __W_ALLOC__( x, w, L )                           # MSB Compen
        #elif L.layer > 0 and L.layer < 3        : out = __W_ALLOC__( x, w, L )                            # Approx w/ W Alloc + I/W Switching
        #elif L.layer == 3                       : out = __W_ALLOC__( x, w, L )                           # MSB Compen

        #if L.layer == 0                         : out = __APPROX__( x, w, L )                           # MSB Compen
        #elif L.layer > 0 and L.layer < 3        : out = __APPROX__( x, w, L )                            # Approx w/ W Alloc + I/W Switching
        #elif L.layer == 3                       : out = __APPROX__( x, w, L )                           # MSB Compen



        if L.layer == 0                         : out = __MSB_COMPEN__( x, w, L )                           # MSB Compen
        elif L.layer > 0 and L.layer < 3        : out = __SWITCHING__( x, w, L )                            # Approx w/ W Alloc + I/W Switching
        elif L.layer == 3                       : out = __MSB_COMPEN__( x, w, L )                           # MSB Compen


        
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        x, w, L = ctx.constant

        if L.BP_quant:
            bp_bw = 8
            int_bw = int(math.log2(torch.abs(grad_output).max()))
            grad_output = custom_precision( grad_output, int_bw, bp_bw-int_bw-1, 'fxp' )

        if L.layer == 3                         : err = torch.mm(grad_output, w.T)                       # Accurate
        elif L.layer<3 and L.layer>0            : err = __MSB_COMPEN_SWITCHING__(grad_output, w.T, L)
        else                                    : err = torch.mm(grad_output, w.T) 

        w_grad = torch.mm(x.T, grad_output)

        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class APPROX(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
        print(f'Approx Inference Mode - Layer {L.layer} !!!') 
        out = __APPROX__(x, w, L)
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        x, w, L = ctx.constant
        
        err = __APPROX__( grad_output, w.T , L )
        w_grad = torch.mm(x.T, grad_output)

        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None


class MSB_COMPEN(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
        out = __MSB_COMPEN__(x, w, L)
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        x, w, L = ctx.constant

        err = __MSB_COMPEN__( grad_output, w.T, L )
        w_grad = torch.mm(x.T, grad_output)
        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None


class SWITCHING(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
        out = __SWITCHING__(x, w, L)
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        x, w, L = ctx.constant

        err = __SWITCHING__( grad_output, w.T, L )
        w_grad = torch.mm(x.T, grad_output)
        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None


class W_ALLOC(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
        out = __W_ALLOC__(x, w, L)
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        x, w, L = ctx.constant

        err = __W_ALLOC__( grad_output, w.T, L )
        w_grad = torch.mm(x.T, grad_output)
        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None




@torch.no_grad()
def __APPROX__(x, w, L):
    print(f'INIT Compressor-based APPROX Comp Layer {L.layer}, scale factor: {L.approx[L.layer]}')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.approx[L.layer]
    return out

@torch.no_grad()
def __MSB_COMPEN__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.msb_compen[L.layer]
    return out

@torch.no_grad()
def __SWITCHING__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.switching[L.layer]
    return out

@torch.no_grad()
def __W_ALLOC__(x, w, L):
    print(f'INIT Compressor-based APPROX+W ALLOC Comp Layer {L.layer}, scale factor: {L.w_alloc[L.layer]}')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.w_alloc[L.layer]
    return out

@torch.no_grad()
def __MSB_COMPEN_W_ALLOC__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.msb_compen_w_alloc[L.layer]
    return out

@torch.no_grad()
def __MSB_COMPEN_SWITCHING__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + x.mean() * L.msb_compen_switching[L.layer]
    return out

