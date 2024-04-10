import torch
import pdb

import lib.gpu_alloc as ga
from torch.autograd import Function

def get_exp_man(x):
    man, exp = torch.frexp(x.abs())
    man, exp = man*2, exp-1 
    
    return exp.float(), man-1

def custom_precision(x, e_i_width, m_f_width, mode):
    if('fxp' in mode):
        if ('sr' in mode):
            return custom_fxp_sr(x, e_i_width, m_f_width)
        else:
            return custom_fxp(x, e_i_width, m_f_width)
    elif(mode=='fp'):
        if(e_i_width==8 and m_f_width==23):
            return x
        return custom_fp(x, e_i_width, m_f_width)
    else:
        print("error in custom_precision!!!")
        exit()

def pow2(x):
    return torch.pow(torch.tensor(2.0, device=ga.device_gpu), x)

def recip(x):
    return torch.tensor( x, device=ga.device_gpu).reciprocal()

def custom_fxp(x, iw , fw): #16bit -> integer = 7, float = 8\\
    iw_exp =pow2(iw)
    fw_exp =pow2(fw)
    fxp_abs_max = iw_exp - recip(fw_exp)

    x = torch.round(x*fw_exp)
    x = x / fw_exp
    return torch.clamp(x, min = -fxp_abs_max, max = fxp_abs_max) # -fxp_abs_max, torch.min(fxp_abs_max, x))


def custom_fxp_sr(x, fxp_i_width, fxp_f_width): #16bit -> integer = 7, float = 8\\
    fxp_i_width_exp =torch.pow(torch.tensor(2.0, device=ga.device_gpu), fxp_i_width)
    fxp_f_width_exp =torch.pow(torch.tensor(2.0, device=ga.device_gpu), fxp_f_width)
    fxp_abs_max = fxp_i_width_exp - torch.tensor(1.0, device=ga.device_gpu)/fxp_f_width_exp

    x = torch.floor(x*fxp_f_width_exp +torch.rand(x.size(), device=ga.device_gpu))
    x = x / fxp_f_width_exp
    return torch.max(-fxp_abs_max, torch.min(fxp_abs_max, x))


@torch.no_grad()
def custom_fp(x, ew, mw):
    mw_exp  = pow2( mw )
    
    e_max   = pow2( ew -1 )-1   # exp range = 128
    fp_max  = pow2( e_max )  # max range = 2^128 -> 1 1111_1111 0x7
    fp_max  = ( 2 - mw_exp.reciprocal() ) * fp_max

    fp_min = pow2( -e_max+1 )      
    du_reci= pow2( mw + e_max.type(torch.float64) - 1  ).type(torch.float64) 

    def log2(x_in):
    	#return torch.div(torch.log(x_in), torch.log(torch.tensor(2.0, device=ga.device_gpu)))
    	return torch.log2(x_in)
    
    #clamp
    x=torch.clamp( x, min=-fp_max, max=fp_max)

    # lower than range  
    mask=(x>0) & (x<fp_min)
    x[mask]=denormal(x[mask], du_reci)
    mask=(x<0) & (x>-fp_min)
    x[mask]=-denormal(-x[mask], du_reci) 
    
    mask=(x==0)
    # mantissa adjust
    sign = torch.sign(x)
    x=x.abs()
    exp = torch.floor(log2(x))
    man = torch.round(mw_exp.type(torch.float64) * torch.mul(x.type(torch.float64) , pow2(-exp.type(torch.float64)))).type( x.type())
    
    x=sign*man/mw_exp *  pow2(exp)

    x[mask]=0
    return x

@torch.no_grad()
def custom_fp_ns(x, ew, mw):
    mw_exp  = pow2( mw )
    
    e_max   = pow2( ew -1 )-1   # exp range = 128
    fp_max  = pow2( e_max )  # max range = 2^128 -> 1 1111_1111 0x7
    fp_max  = ( 2 - mw_exp.reciprocal() ) * fp_max

    fp_min = pow2( -e_max+1 )      
    #du_reci= pow2( mw + e_max.type(torch.float64) - 1  ).type(torch.float64) 

    def log2(x_in):
    	#return torch.div(torch.log(x_in), torch.log(torch.tensor(2.0, device=ga.device_gpu)))
    	return torch.log2(x_in)
    
    #clamp
    x=torch.clamp( x, min=-fp_max, max=fp_max)

    # lower than range  
    #mask=(x>0) & (x<fp_min)
    #x[mask]=denormal(x[mask], du_reci)
    #mask=(x<0) & (x>-fp_min)
    #x[mask]=-denormal(-x[mask], du_reci) 
    
    mask=(x<fp_min) & (x>-fp_min)
    # mantissa adjust
    sign = torch.sign(x)
    x=x.abs()
    exp = torch.floor(log2(x))
    man = torch.round(mw_exp.type(torch.float64) * torch.mul(x.type(torch.float64) , pow2(-exp.type(torch.float64)))).type( x.type())
    
    x=sign*man/mw_exp *  pow2(exp)

    x[mask]=0
    return x

def denormal(x, denormal_unit_reci):
    return (torch.round(x.type(torch.float64) * denormal_unit_reci)/denormal_unit_reci).type(x.type())
   
@torch.no_grad()
def fp(x, mode=-1):
    
    # Precision
    if   mode < 0   : return x.type(torch.float32)   
    if   mode == 0   : return x.type(torch.float16)   
    elif mode == 8   : return custom_fp(x,4,3)
    elif mode == 9   : return custom_fp(x,5,3)
    elif mode == 16  : return custom_fp(x,5,10)
    elif mode == 17  : return custom_fp(x,6,9)
    elif mode == 18  : return custom_fp(x,8,7)
    elif mode == 32  : return custom_fp(x,8,23)

    ## NS 
    elif mode == 8.1     : return custom_fp_ns(x,4,3)
    elif mode == 16.1    : return custom_fp_ns(x,5,10)
    elif mode == 17.1    : return custom_fp_ns(x,6,9)
    elif mode == 18.1    : return custom_fp_ns(x,8,7)
    elif mode == 32.1   :   return custom_fp_ns(x,8,23) 



class Change_precision_FF_function(Function):
    @staticmethod
    def forward(ctx, x, e_i, m_f, fp_fxp ):
        ctx.constant = (e_i, m_f, fp_fxp)
        out = custom_precision(x, e_i, m_f, fp_fxp)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        e_i_width, m_f_width, fp_fxp = ctx.constant
        out = custom_precision(grad_output, 5, 10, 'fp')
        #out = grad_output
        return out, None, None, None, None

#class Change_precision_W_function(Function):
#    @staticmethod
#    def forward(ctx, x, e_i, m_f, fp_fxp ):
#        ctx.constant = (e_i, m_f, fp_fxp)
#        out = custom_precision(x, e_i, m_f, fp_fxp)
#        return out
#    @staticmethod
#    def backward(ctx, grad_output):
#        e_i_width, m_f_width, fp_fxp = ctx.constant
#        #out = custom_precision(grad_output, o_e_i_width, o_m_f_width, o_fp_fxp)
#        out = grad_output
#        return out, None, None, None, None

class Change_precision_ENC_function(Function):
    @staticmethod
    def forward(ctx, x, e_i, m_f, fp_fxp ):
        ctx.constant = (e_i, m_f, fp_fxp)
        out = custom_precision(x, e_i, m_f, fp_fxp)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        e_i_width, m_f_width, fp_fxp = ctx.constant
        out = custom_precision(grad_output, 5, 10, 'fp')
        #out = grad_output
        return out, None, None, None, None

