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

class CFC(nn.Linear):
    def __init__(self, in_features, out_features, bias, training=False, inf_accurate=False, bp_accurate=False, eval_ich_range=False, inf_approx=False, inf_msb=False, inf_switching=False, inf_w_alloc=False, layer_wise_m0=False, msb_w_alloc_switch=False):
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
        self.layer_wise_m0 = layer_wise_m0
        self.msb_w_alloc_switch = msb_w_alloc_switch
        self.ich_flag = False
        self.inf_quant = False
        self.bp_quant = False
        self.layer = 0

        self.approx = [0.501, 0.553, 0.571, 0.573]                          # Approx
        self.w_alloc = [0.554, 0.584, 0.519, 0.527]                         # Approx w/ Weight Alloc
        self.switching = [0.4, 0.421, 0.438, 0.453]                         # Approx w/ Weight Alloc + I/W Switching
        self.msb_compen = [0.295, 0.207, 0.211, 0.223]                      # Carry Compensation
        self.msb_compen_w_alloc = [0.175, 0.155, 0.153, 0.153]              # Carry w/ Weight Alloc
        self.msb_compen_switching = [0.0055, 0.0092, 0.0091, 0.0027]        # Carry w/ Weight Alloc + I/W Switching
        #self.approx = [0.002689313, 0.007436308, 0.03508, 0.389]
        #self.weight_shuffle = [0.001648244, 0.004663158, 0.021886315, 0.323520482]
        #self.switching = [0.001068843, 0.004613975, 0.016299255, 0.29205894]
        #self.msb_compen = [0.008, 0.0022, 0.0121, 0.2175]



    def forward( self, x ):
        #if self.eval_ich_range : eval_ich_range(x)
        

        if self.training                        : out = Accurate.apply( x, self.weight.T, self )
        elif self.inf_accurate                  : out = Accurate.apply( x, self.weight.T, self )
        elif self.inf_approx                    : out = APPROX.apply( x, self.weight.T , self)
        elif self.inf_msb                       : out = MSB_COMPEN.apply( x, self.weight.T, self )
        elif self.inf_switching                 : out = SWITCHING.apply( x, self.weight.T, self )
        elif self.inf_w_alloc                   : out = W_ALLOC.apply( x, self.weight.T, self )
        elif self.layer_wise_m0                 : out = LAYER_WISE_V0.apply( x, self.weight.T, self )
        elif self.msb_w_alloc_switch            : out = MSB_COMPEN_W_ALLOC_SWITCH.apply( x, self.weight.T, self )
        else                                    : out = Accurate.apply( x, self.weight.T, self )
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Accurate(Function):
    @staticmethod
    def forward( ctx, x, w , L ):

        if L.inf_quant:
            int_bw = int(math.log2(torch.abs(x).max()))
            x = custom_precision( x, 4, 0, 'fxp' )

            x_nonz = x[x!=0]
            hist = torch.histc(x_nonz.reshape(-1).cuda(), bins=100) 

            maxi = hist.max()
            maxi_bmap = hist == maxi
            maxi_idx = torch.nonzero(maxi_bmap)
           
            if maxi_idx < hist.shape[0]-1:
                for bit in range (1, 4):
                    b = 2**(bit-1) -1

                    indexing = hist[maxi_idx : maxi_idx+2*b+1]
                    print(f'{bit}: {indexing.sum()/hist.sum()}, offloading ich: {128*(1- indexing.sum()/hist.sum() )}')

                    print(f'===================================================================================================================')


            bins = torch.linspace(x_nonz.min(), x_nonz.max(), 100)
            hist = torch.cat( [bins.reshape(1,-1), hist.reshape(1,-1)], dim=0 )
            np.savetxt('./result/inference_input_l'+str(L.layer)+'.csv', hist.detach().cpu().numpy(), delimiter=',')

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


        if L.bp_quant:
            int_bw = int(math.log2(torch.abs(grad_output).max()))
            grad_output = custom_precision( grad_output, 8, 0, 'fxp' )

            grad_nonz = grad_output[grad_output!=0]
            hist = torch.histc(grad_nonz.reshape(-1).cuda(), bins=100) 

            maxi = hist.max()
            maxi_bmap = hist == maxi
            maxi_idx = torch.nonzero(maxi_bmap)
           
            if maxi_idx < hist.shape[0]-1 and maxi_idx > 0:
                for bit in range (3, 8):
                    b = 2**(bit-1) -1

                    indexing = hist[maxi_idx-b : maxi_idx+b+1]
                    print(f'{bit}: {indexing.shape} {indexing.sum()/hist.sum()}, offloading ich: {128*(1- indexing.sum()/hist.sum() )}')


            bins = torch.linspace(grad_output.min(),grad_output.max(), 100)
            hist = torch.cat( [bins.reshape(1,-1), hist.reshape(1,-1)], dim=0 )
            np.savetxt('./result/bp_error_l'+str(L.layer)+'.csv', hist.detach().cpu().numpy(), delimiter=',')
            #L.writer.UpdateData(data=grad_output.reshape(-1).detach().cpu().numpy())
            #L.writer.PngHistPlot( file_path='./result/bp_result_l'+str(L.layer)+'.png', title=f'Layer {L.layer} Err Distribution', xlabel=f'Err Value', ylabel='Num', bins=20 )


        if L.ich_flag : np.savetxt('./result/backward_result_l'+str(L.layer)+'.csv', grad_output.detach().cpu().numpy(), delimiter=',')

        err = torch.mm( grad_output, w.T )


        w_grad = torch.mm( x.T.type(torch.float32), grad_output.type(torch.float32) )
        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class MSB_COMPEN_W_ALLOC_SWITCH(Function):      # GRAD
    @staticmethod
    def forward( ctx, x, w, L ): 
        print(f'MSB Compensation w/ Weight Alloc & I/W Switching Inference - Layer {L.layer} !!!') 
        out = __MSB_COMPEN_SWITCHING__( x, w, L ) 
        ctx.constant = ( x, w, L )
        return out
    def backward( ctx, grad_output ):
        x, w, L = ctx.constant 
        print(f'MSB Compensation w/ Weight Alloc & I/W Switching Backward  - Layer {L.layer} !!!') 

        err = __MSB_COMPEN_SWITCHING__(grad_output, w.T, L)                       # Accurate

        w_grad = torch.mm(x.T, grad_output)

        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class LAYER_WISE_V0(Function):      # a_star
    @staticmethod
    def forward( ctx, x, w, L ):
        print(f'A Star Inference Mode - Layer {L.layer} !!!') 
        if L.layer == 0                         : out = __MSB_COMPEN__( x, w, L )                           # MSB Compen
        elif L.layer > 0 and L.layer < 3        : out = __SWITCHING__( x, w, L )                            # Approx w/ W Alloc + I/W Switching
        elif L.layer == 3                       : out = __MSB_COMPEN__( x, w, L )                           # MSB Compen
        
        ctx.constant = ( x, w, L )
        return out

    def backward( ctx, grad_output ):
        x, w, L = ctx.constant

        if L.layer == 3                         : err = torch.mm(grad_output, w.T)                       # Accurate
        elif L.layer<3 and L.layer>0            : err = __MSB_COMPEN_SWITCHING__(grad_output, w.T, L)
        else                                    : err = torch.mm(grad_output, w.T) 

        w_grad = torch.mm(x.T, grad_output)

        
        return err.type( torch.float32 ), w_grad.type( torch.float32 ), None

class APPROX(Function):
    @staticmethod
    def forward( ctx, x, w, L ):
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
    #print(f'INIT Compressor-based APPROX Comp Layer {L.layer}')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.approx[L.layer]           # --a_star_approx_scaling
    # print(f'dh a star approx scaling factor is { opt.XXXX}')
    return out

@torch.no_grad()
def __MSB_COMPEN__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.msb_compen[L.layer]       # --a_star_msb_scaling
    return out

@torch.no_grad()
def __SWITCHING__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
    out = torch.mm( x, w )
    out = out + torch.abs(out).mean() * L.switching[L.layer]
    return out

@torch.no_grad()
def __W_ALLOC__(x, w, L):
    #print(f'INIT Compressor-based MSB Compen Comp')
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
    out = out + x.mean() * L.msb_compen_switching[L.layer] # argument --bp_scaling 0.01
    return out

@torch.no_grad()
def eval_ich_range(x):
    pdb.set_trace()
