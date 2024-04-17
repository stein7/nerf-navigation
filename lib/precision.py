import torch
import pdb

import lib.gpu_alloc as ga

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

def custom_fxp(x, fxp_i_width, fxp_f_width): #16bit -> integer = 7, float = 8\\
    fxp_i_width_exp =torch.pow(torch.tensor(2.0, device=ga.device_gpu), fxp_i_width)
    fxp_f_width_exp =torch.pow(torch.tensor(2.0, device=ga.device_gpu), fxp_f_width)
    fxp_abs_max = fxp_i_width_exp - torch.tensor(1.0 , device=ga.device_gpu)/fxp_f_width_exp

    x = torch.round(x*fxp_f_width_exp)
    x = x / fxp_f_width_exp
    return torch.max(-fxp_abs_max, torch.min(fxp_abs_max, x))

def custom_fxp_sr(x, fxp_i_width, fxp_f_width): #16bit -> integer = 7, float = 8\\
    fxp_i_width_exp =torch.pow(torch.tensor(2.0, device=ga.device_gpu), fxp_i_width)
    fxp_f_width_exp =torch.pow(torch.tensor(2.0, device=ga.device_gpu), fxp_f_width)
    fxp_abs_max = fxp_i_width_exp - torch.tensor(1.0, device=ga.device_gpu)/fxp_f_width_exp

    x = torch.floor(x*fxp_f_width_exp +torch.rand(x.size(), device=ga.device_gpu))
    x = x / fxp_f_width_exp
    return torch.max(-fxp_abs_max, torch.min(fxp_abs_max, x))


def custom_fp(x, fp_e_width, fp_m_width):
    fp_e_width_exp = torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_e_width)
    fp_m_width_exp=torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_m_width)
    fp_exp_range = torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_e_width) / 2  # exp range = 128
    fp_max_range = torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_exp_range)  # max range = 2^128 -> 1 1111_1111 0x7
    fp_max_data = (1 + (torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_m_width) - 1) / torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_m_width)) * fp_max_range
    fp_min_data = torch.pow(torch.tensor(2.0, device=ga.device_gpu), -fp_exp_range+2)
    denormal_unit_reci=torch.pow(torch.tensor(2.0, device=ga.device_gpu), fp_m_width+fp_exp_range-2)

    def log2(x_in):
    	return torch.div(torch.log(x_in), torch.log(torch.tensor(2.0, device=ga.device_gpu)))
    #exp filtering
    x=torch.min(fp_max_data, torch.max(-fp_max_data, x))
    mask=(x>0) & (x<fp_min_data)
    x[mask]=denormal(x[mask], denormal_unit_reci)
    mask=(x<0) & (x>-fp_min_data)
    x[mask]=-denormal(-x[mask], denormal_unit_reci)

    mask=(x==0)
    # mantissa adjust
    sign = torch.sign(x)
    x=x*sign
    exp = torch.floor(log2(x))
    man = torch.round(fp_m_width_exp * torch.mul(x, torch.pow(torch.tensor(2.0, device=ga.device_gpu), -exp)))
    x=sign*man/fp_m_width_exp * torch.pow(torch.tensor(2.0, device=ga.device_gpu), exp)
    x[mask]=0
    return x
    
def denormal(x, denormal_unit_reci):
    return torch.round(x*denormal_unit_reci)/denormal_unit_reci

