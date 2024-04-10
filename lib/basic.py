import os
import pdb
import numpy as np

import cv2
import scipy

import torch

import time
import datetime
import sys

import inspect, re
import scipy.io as io

from lib.utils.utils import progress_bar
from lib.utils.utilfunc import *
#from lib.utils.memory_debug import *

from lib.precision import custom_precision
from lib.int_adapt import *
from lib.LDPS import *
from lib.dataset import prepare_dataset
from lib.loss_function import *

from torch.autograd import Variable

def int_init(net, trainloader, device, optimizer, criterion, loss_scale, config_dict):
    net.train()
    save_data = False
    save_num = 0
    loss_temp = 0
    correct = 0
    total = 0
    loss_list = list()

    gpu_num, pw_fp_fxp, w_fp_fxp, \
    a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp, \
    g_fp_fxp, w_int, w_slice, \
    a_int, a_slice, e_int, e_slice, \
    ao_int, ao_slice, eo_int, eo_slice, \
    w_ovf, w_sp, a_ovf, a_sp, e_ovf, e_sp, ao_ovf, ao_sp, eo_ovf, eo_sp, \
    g_int, g_slice, pw_int, pw_slice, LDPS_config, activation = load_config(config_dict)
    w_check = False
    a_check = False
    e_check = False
    ao_check = False
    eo_check = False
    
    rep_num = 0
    while ((not(w_check & a_check & e_check & ao_check & eo_check)) & (rep_num < 30)):

        w_int, w_ovf, w_sp, \
        a_int, a_ovf, a_sp, \
        e_int, e_ovf, e_sp, \
        ao_int, ao_ovf, ao_sp, \
        eo_int, eo_ovf, eo_sp = int_adapt(w_int,    w_ovf,    w_sp, \
                                       a_int,    a_ovf,    a_sp, \
                                       e_int,    e_ovf,    e_sp, \
                                       ao_int,   ao_ovf,   ao_sp, \
                                       eo_int,   eo_ovf,   eo_sp, \
                                       w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
        this_is_bias = False
        old_params, old_bias=[], []
        index, index_b = 0, 0
        for name, param in net.named_parameters():
            if param.dim() != 1:
                w_ei=w_int[index]
                w_mf=w_slice[index]-w_int[index] -1
                w_bw = w_slice[index]
                old_params.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                w_ovf[index], w_sp[index] = ovf_sp(old_params[-1], param.data,w_bw,  w_ei, w_ovf[index], w_sp[index], w_fp_fxp)
                this_is_bias = True
                param.data=old_params[-1]
                index=index+1
            elif 'EPO_status' in name:
                this_is_bias = False
            elif 'EP_status' in name:
                this_is_bias = False
            elif this_is_bias and not('bw_param' in name):
                old_bias.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                w_ovf[index-1], w_sp[index-1] = ovf_sp(old_bias[-1], param.data, w_bw, w_ei, w_ovf[index-1], w_sp[index-1], w_fp_fxp)
                this_is_bias = False
                param.data=old_bias[-1]
                index_b=index_b+1
            if 'bw_param' in name:
                param.data = torch.tensor([w_int[index-1], gpu_num])
            


        w_check, a_check, e_check, ao_check, eo_check \
        = check_int(w_ovf, w_sp, a_ovf, a_sp, e_ovf, e_sp, ao_ovf, ao_sp, eo_ovf, eo_sp, \
                    w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
        
        print('w_int    = ' + str(w_int       )) #+ str(w_check))
        rep_num = rep_num + 1

    w_check = False
    a_check = False
    e_check = False
    ao_check = False
    eo_check = False
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):  
        rep_num = 0
        while ((not(w_check & a_check & e_check & ao_check & eo_check)) & (rep_num < 30)):
            #---Weight & Activation & Error Precision Handle---#
            w_int, w_ovf, w_sp, \
            a_int, a_ovf, a_sp, \
            e_int, e_ovf, e_sp, \
            ao_int, ao_ovf, ao_sp, \
            eo_int, eo_ovf, eo_sp = int_adapt(w_int,    w_ovf,    w_sp, \
                                           a_int,    a_ovf,    a_sp, \
                                           e_int,    e_ovf,    e_sp, \
                                           ao_int,   ao_ovf,   ao_sp, \
                                           eo_int,   eo_ovf,   eo_sp, \
                                           w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
            
            old_params, old_bias=[], []
            index, index_b, index_a, index_ao, index_e, index_eo = 0, 0, 0, 0, 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    w_ei=w_int[index]
                    w_mf=w_slice[index]-w_int[index] -1
                    w_bw = w_slice[index]
                    old_params.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                    w_ovf[index], w_sp[index] = ovf_sp(old_params[-1], param.data,w_bw, w_ei, w_ovf[index], w_sp[index], w_fp_fxp)
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                    this_is_bias = False
                    index_a = index_a + 1
                elif 'FFO_status' in name:
                    param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                    this_is_bias = False
                    index_ao = index_ao + 1
                elif 'EP_status' in name:
                    param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                    this_is_bias = False
                    index_e = index_e + 1
                elif 'EPO_status' in name:
                    param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                    this_is_bias = False
                    index_eo = index_eo + 1
                elif this_is_bias and not('bw_param' in name):
                    old_bias.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                    w_ovf[index-1], w_sp[index-1] = ovf_sp(old_bias[-1], param.data, w_bw,w_ei, w_ovf[index-1], w_sp[index-1], w_fp_fxp)
                    this_is_bias = False
                    index_b=index_b+1

            #---Feed Forward---#
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            #---Error Propagation---#
            loss = loss_scale*criterion(outputs, targets)
            #loss = loss_scale*LabelSmoothingLoss(outputs, targets)
            loss.backward()
            
            #---Overflow & Surplus Detection + Primal Weight Recovery---#
            index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    a_ovf[index_a] = float(param.data[2])
                    a_sp[index_a] = float(param.data[3])
                    this_is_bias = False
                    index_a = index_a + 1
                elif 'FFO_status' in name:
                    ao_ovf[index_ao] = float(param.data[2])
                    ao_sp[index_ao] = float(param.data[3])
                    this_is_bias = False
                    index_ao = index_ao + 1
                elif 'EP_status' in name:
                    e_ovf[index_e] = float(param.data[2])
                    e_sp[index_e] = float(param.data[3])
                    this_is_bias = False
                    index_e = index_e + 1
                elif 'EPO_status' in name:
                    eo_ovf[index_eo] = float(param.data[2])
                    eo_sp[index_eo] = float(param.data[3])
                    this_is_bias = False
                    index_eo = index_eo + 1
                elif this_is_bias and not('bw_param' in name):
                    param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = False
                    index_b=index_b+1

            w_check, a_check, e_check, ao_check, eo_check \
            = check_int(w_ovf, w_sp, a_ovf, a_sp, e_ovf, e_sp, ao_ovf, ao_sp, eo_ovf, eo_sp, \
                        w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
            
    
            print('a_int    = ' + str(a_int       )) #+ str(a_check))
            print('e_int    = ' + str(e_int       )) #+ str(e_check))
            print('ao_int   = ' + str(ao_int      )) #+ str(a_check))
            print('eo_int   = ' + str(eo_int      )) #+ str(e_check))
            rep_num = rep_num + 1

        break

    config_dict = save_config(w_int, w_slice, \
                              a_int, a_slice, e_int, e_slice, \
                              ao_int, ao_slice, eo_int, eo_slice, \
                              w_ovf, w_sp, \
                              a_ovf, a_sp, e_ovf, e_sp, \
                              ao_ovf, ao_sp, eo_ovf, eo_sp, \
                              config_dict)

    return config_dict


def LAPS(net, trainloader, device,optimizer, criterion, epoch, g_scale , loss_scale, config_dict, LAPS_Controller):
    print("=============================== Bit Decision ==============================")
    net.train()
    stop_list = list()
    LAPS_Controller.do_save = True

    fsm_period = {
        'v2'            : 4,
        'v2.1'          : 4,
        'v2_theta'      : 4, 
        'fsm_5'         : 3,
        'fsm_5.1'       : 3,
        'const'         : 1,
        'ref_compare'   : 1
    }

    folder = config_dict['opt'].LDPS_folder

    gpu_num, pw_fp_fxp, w_fp_fxp, \
    a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp, \
    g_fp_fxp, w_int, w_slice, \
    a_int, a_slice, e_int, e_slice, \
    ao_int, ao_slice, eo_int, eo_slice, \
    w_ovf, w_sp, \
    a_ovf, a_sp, e_ovf, e_sp, \
    ao_ovf, ao_sp, eo_ovf, eo_sp, \
    g_int, g_slice, pw_int, pw_slice, LDPS_config, activation = load_config(config_dict)

    period = fsm_period['v2' if 'v2' in LDPS_config else LDPS_config]

    if 'v2' in LDPS_config : 
        config_dict['W_compare_result'] = torch.zeros( len(w_int) , period +1  )
        config_dict['A_compare_result'] = torch.zeros( len(w_int) , period +1  )
        config_dict['G_compare_result'] = torch.zeros( len(w_int) , period +1  )
        config_dict['EP_compare_result'] = torch.zeros( len(w_int) , period +1  )
    config_dict['F_state'], config_dict['B_state'] = ['start'] * len(w_int), ['start']* len(w_int)
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):  
        #---Weight & Activation & Error Precision Handle---#
        w_int, w_ovf, w_sp, \
        a_int, a_ovf, a_sp, \
        e_int, e_ovf, e_sp, \
        ao_int, ao_ovf, ao_sp, \
        eo_int, eo_ovf, eo_sp = int_adapt(w_int,    w_ovf,    w_sp, \
                                       a_int,    a_ovf,    a_sp, \
                                       e_int,    e_ovf,    e_sp, \
                                       ao_int,   ao_ovf,   ao_sp, \
                                       eo_int,   eo_ovf,   eo_sp, \
                                       w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
       
        config_dict = save_config(w_int, w_slice, \
                              a_int, a_slice, e_int, e_slice, \
                              ao_int, ao_slice, eo_int, eo_slice, \
                              w_ovf, w_sp, \
                              a_ovf, a_sp, e_ovf, e_sp, \
                              ao_ovf, ao_sp, eo_ovf, eo_sp, \
                              config_dict)
     
        # bit increase before inferecne 
        w_change, a_change, e_change, bw_change, is_LDPS = LDPS_control_1(batch_idx % period , w_fp_fxp[:4], config_dict)

        if is_LDPS : 
            w_slice = config_change(w_slice, LDPS_config , w_change )
            a_slice = config_change(a_slice, LDPS_config , e_change ) 
            e_slice = config_change(e_slice, LDPS_config , a_change )
            if not (config_dict['algo'] == 'None') : 
                origin_bw = bw_slice_change(net, [0]*1000, LC = LDPS_config)
                bw_slice = bw_slice_change(net, bw_change, LC = LDPS_config, is_print = True) # increase all bw slice length
                config_dict['bw_change'] = [ob - b for b, ob in zip(bw_slice,origin_bw)]
        
            storing_control(net, 'on')

        if 'ref' in LDPS_config :  w_slice, a_slice, e_slice, w_fp_fxp, a_fp_fxp, e_fp_fxp, net, config_dict = ref_config_change(net, config_dict, 'ref')

        
        
        old_params, old_bias=[], []
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        this_is_bias = False
        for name, param in net.named_parameters():
            if param.dim() != 1:
                w_ei=w_int[index]
                w_mf=w_slice[index]-w_int[index] -1
                w_bw = w_slice[index]
                old_params.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                w_ovf[index], w_sp[index] = ovf_sp(old_params[-1], param.data,w_bw, w_ei, w_ovf[index], w_sp[index], w_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                old_bias.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                w_ovf[index-1], w_sp[index-1] = ovf_sp(old_bias[-1], param.data,w_bw,  w_ei, w_ovf[index-1], w_sp[index-1], w_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
            if 'bw_param' in name :
                param.data = torch.tensor( [ w_int[index-1] , gpu_num])
                #print(name)
       
        #---Feed Forward---#
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        #---Error Propagation---#
        loss = loss_scale*criterion(outputs, targets)
        loss.backward()

        if 'ref' in LDPS_config :  w_slice, a_slice, e_slice, w_fp_fxp, a_fp_fxp, e_fp_fxp, net, config_dict = ref_config_change(net, config_dict, 'org')
        
        

        if is_LDPS :
            #input_list, output_list = get_front_inout(net)
            config_dict , FF_stat_list = LDPS_FF(net, config_dict, epoch, LDPS_config, LAPS_Controller)
            w_slice = config_dict['w_slice']
            a_slice = config_dict['a_slice']
            if 'v2' in LDPS_config : 
                for i in range(len(w_slice)) : config_dict['W_compare_result'][i][batch_idx % period] = config_dict['temp_compare_result_w'][i]
                for i in range(len(w_slice)) : config_dict['A_compare_result'][i][batch_idx % period] = config_dict['temp_compare_result_a'][i]
        
        

        if is_LDPS  :
            config_dict, EP_stat_list = LDPS_EP(net, config_dict, epoch, LDPS_config, LAPS_Controller)
            e_slice = config_dict['e_slice']
            if 'v2' in LDPS_config :
                for i in range(len(e_slice)-1) : config_dict['EP_compare_result'][i][batch_idx % period] = config_dict['temp_compare_result'][i]


        #---Gradient Precision Handle---#
        #---Overflow & Surplus Detection + Primal Weight Recovery---#
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        this_is_bias = False
        for name, param in net.named_parameters():
            if param.dim() != 1:
                if ('dfxp' in g_fp_fxp) :
                    param.grad=custom_precision(param.grad, w_int[index]+g_scale, g_slice-w_int[index]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                torch.save(param.grad, './LDPS_data/'+folder+'/grad_'+str(index).zfill(3)+'pt')
                param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index]+1, pw_slice-w_int[index]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                a_ovf[index_a] = float(param.data[2])
                a_sp[index_a] = float(param.data[3])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                ao_ovf[index_ao] = float(param.data[2])
                ao_sp[index_ao] = float(param.data[3])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                e_ovf[index_e] = float(param.data[2])
                e_sp[index_e] = float(param.data[3])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                eo_ovf[index_eo] = float(param.data[2])
                eo_sp[index_eo] = float(param.data[3])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                if ('dfxp' in g_fp_fxp):
                    param.grad=custom_precision(param.grad, w_int[index-1] + g_scale, g_slice-w_int[index-1]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index-1] + 1, pw_slice-w_int[index-1]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
        
        
     
        #low-bit bersion gradient computation
        old_params, old_bias=[], []
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        this_is_bias = False
        for name, param in net.named_parameters():
            if param.dim() != 1:
                w_ei=w_int[index]
                w_mf=w_slice[index]-w_int[index] -1
                w_bw = w_slice[index]
                old_params.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                w_ovf[index], w_sp[index] = ovf_sp(old_params[-1], param.data,w_bw, w_ei, w_ovf[index], w_sp[index], w_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                old_bias.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                w_ovf[index-1], w_sp[index-1] = ovf_sp(old_bias[-1], param.data,w_bw,  w_ei, w_ovf[index-1], w_sp[index-1], w_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
            if 'bw_param' in name :
                param.data = torch.tensor( [ w_int[index-1] , gpu_num])
                #print(name)
        #---Feed Forward---#

        optimizer.zero_grad()
        outputs = net(inputs)
        loss_g = loss_scale*criterion(outputs, targets)
        loss_g.backward()



        #---Gradient Precision Handle---#
        #---Overflow & Surplus Detection + Primal Weight Recovery---#
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        this_is_bias = False
        temp_g_compare = []
        for name, param in net.named_parameters():
            if param.dim() != 1:
                if ('dfxp' in g_fp_fxp) :
                    param.grad=custom_precision(param.grad, w_int[index]+g_scale, g_slice-w_int[index]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                
                temp_g_compare.append( LDPS_GR(config_dict, torch.load('./LDPS_data/'+folder+'/grad_'+str(index).zfill(3)+'pt'),  param.grad, index, LAPS_Controller))

                param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index]+1, pw_slice-w_int[index]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                a_ovf[index_a] = float(param.data[2])
                a_sp[index_a] = float(param.data[3])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                ao_ovf[index_ao] = float(param.data[2])
                ao_sp[index_ao] = float(param.data[3])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                e_ovf[index_e] = float(param.data[2])
                e_sp[index_e] = float(param.data[3])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                eo_ovf[index_eo] = float(param.data[2])
                eo_sp[index_eo] = float(param.data[3])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                if ('dfxp' in g_fp_fxp):
                    param.grad=custom_precision(param.grad, w_int[index-1] + g_scale, g_slice-w_int[index-1]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index-1] + 1, pw_slice-w_int[index-1]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = False
                index_b=index_b+1

        optimizer.zero_grad()
        if 'v2' in LDPS_config :
            for i in range(len(e_slice)) : 
                config_dict['G_compare_result'][i][batch_idx % period] = temp_g_compare[i]
        if ('v2' in LDPS_config) and ((batch_idx +1) % period ==0) :
            config_dict = v2_decision(config_dict)
            if not(config_dict['algo'] == 'None' ) : bw_slice_change(net, config_dict['bw_update_list'])
        
        LAPS_Controller.stop_update()
        stop_list.append(LAPS_Controller.stop)
        #print(LAPS_Controller.stable_F, LAPS_Controller.stable_B, stop_list)
        if (batch_idx +1) % period ==0 : 
            global_stop = True
            for i in stop_list : global_stop = global_stop and i 
            print(batch_idx)
            if global_stop or (batch_idx == period * 5 - 1  ) :
                print("=========================== Bit Decided at " + str(  int((batch_idx+1)/period)  ) + " ===========================")
                return config_dict
            else : stop_list = list()


def train(net, trainloader, device, optimizer, criterion, epoch, g_scale, loss_scale, break_train, config_dict):
    net.train()
    save_data = False
    save_num = 0
    loss_temp = 0
    correct = 0
    total = 0
    loss_list = list()

    gpu_num, pw_fp_fxp, w_fp_fxp, \
    a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp, \
    g_fp_fxp, w_int, w_slice, \
    a_int, a_slice, e_int, e_slice, \
    ao_int, ao_slice, eo_int, eo_slice, \
    w_ovf, w_sp, \
    a_ovf, a_sp, e_ovf, e_sp, \
    ao_ovf, ao_sp, eo_ovf, eo_sp, \
    g_int, g_slice, pw_int, pw_slice, LDPS_config, activation = load_config(config_dict)
    bw_slice_change(net , mode = [0] * 1000 , LC = LDPS_config, is_print = True)

    #if (epoch != 0) & (epoch % 2 == 0):
    #    if ('dfxp' in w_fp_fxp):
    #        for list_n in range(1, len(w_slice)-1):
    #            w_slice[list_n] = max(w_slice[list_n] - 2, 2)
    #    if ('dfxp' in a_fp_fxp):
    #        for list_n in range(1, len(a_slice)-1):
    #            a_slice[list_n] = max(a_slice[list_n] - 2, 2)
    #    if ('dfxp' in e_fp_fxp):
    #        for list_n in range(1, len(e_slice)-1):
    #            e_slice[list_n] = max(e_slice[list_n] - 2, 6)
    #    if ('dfxp' in ao_fp_fxp):
    #        for list_n in range(1, len(ao_slice)-1):
    #            ao_slice[list_n] = max(ao_slice[list_n] - 2, 1)
    #    if ('dfxp' in eo_fp_fxp):
    #        for list_n in range(1, len(eo_slice)-1):
    #            eo_slice[list_n] = max(eo_slice[list_n] - 2, 1)
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):  
        #---Weight & Activation & Error Precision Handle---#
        w_int, w_ovf, w_sp, \
        a_int, a_ovf, a_sp, \
        e_int, e_ovf, e_sp, \
        ao_int, ao_ovf, ao_sp, \
        eo_int, eo_ovf, eo_sp = int_adapt(w_int,    w_ovf,    w_sp, \
                                       a_int,    a_ovf,    a_sp, \
                                       e_int,    e_ovf,    e_sp, \
                                       ao_int,   ao_ovf,   ao_sp, \
                                       eo_int,   eo_ovf,   eo_sp, \
                                       w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
       
        config_dict = save_config(w_int, w_slice, \
                              a_int, a_slice, e_int, e_slice, \
                              ao_int, ao_slice, eo_int, eo_slice, \
                              w_ovf, w_sp, \
                              a_ovf, a_sp, e_ovf, e_sp, \
                              ao_ovf, ao_sp, eo_ovf, eo_sp, \
                              config_dict)

        old_params, old_bias=[], []
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        this_is_bias = False
        for name, param in net.named_parameters():
            if param.dim() != 1:
                w_ei=w_int[index]
                w_mf=w_slice[index]-w_int[index] -1
                w_bw = w_slice[index]
                old_params.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                w_ovf[index], w_sp[index] = ovf_sp(old_params[-1], param.data,w_bw, w_ei, w_ovf[index], w_sp[index], w_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                old_bias.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                w_ovf[index-1], w_sp[index-1] = ovf_sp(old_bias[-1], param.data,w_bw,  w_ei, w_ovf[index-1], w_sp[index-1], w_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
            if 'bw_param' in name :
                param.data = torch.tensor( [ w_int[index-1] , gpu_num])
                #print(name)

        #---Feed Forward---#
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        #---Error Propagation---#
        loss = loss_scale*criterion(outputs, targets)
        #loss = loss_scale*LabelSmoothingLoss(outputs, targets)
        loss.backward()

        #---Gradient Precision Handle---#
        #---Overflow & Surplus Detection + Primal Weight Recovery---#
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        for name, param in net.named_parameters():
            if param.dim() != 1:
                if ('dfxp' in g_fp_fxp) :
                    param.grad=custom_precision(param.grad, w_int[index]+g_scale, g_slice-w_int[index]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                #save_mat(save_data & (batch_idx >= save_num), param.grad.cpu().numpy(), 'grad', index)
                param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index]+1, pw_slice-w_int[index]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                a_ovf[index_a] = float(param.data[2])
                a_sp[index_a] = float(param.data[3])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                ao_ovf[index_ao] = float(param.data[2])
                ao_sp[index_ao] = float(param.data[3])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                e_ovf[index_e] = float(param.data[2])
                e_sp[index_e] = float(param.data[3])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                eo_ovf[index_eo] = float(param.data[2])
                eo_sp[index_eo] = float(param.data[3])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                if ('dfxp' in g_fp_fxp):
                    param.grad=custom_precision(param.grad, w_int[index-1] + g_scale, g_slice-w_int[index-1]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index-1] + 1, pw_slice-w_int[index-1]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
                

        if (break_train):
            loss_temp = 0
            total = 1
            correct = 0
            break
        else:
            #---Optimizer (w Update)---#
            optimizer.step()
            loss_temp += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_temp/(batch_idx+1), 100.*correct/total, correct, total))
    

    config_dict = save_config(w_int, w_slice, \
                              a_int, a_slice, e_int, e_slice, \
                              ao_int, ao_slice, eo_int, eo_slice, \
                              w_ovf, w_sp, \
                              a_ovf, a_sp, e_ovf, e_sp, \
                              ao_ovf, ao_sp, eo_ovf, eo_sp, \
                              config_dict)

    return 100.*correct/total, loss_temp/(batch_idx+1), config_dict , loss_list

def test(net, testloader, device, optimizer, criterion, epoch, config_dict):
    net.eval()
    loss_temp = 0
    correct = 0
    total = 0

    gpu_num     ,pw_fp_fxp      ,w_fp_fxp       ,a_fp_fxp   ,\
    e_fp_fxp    ,ao_fp_fxp      ,eo_fp_fxp      ,g_fp_fxp   ,\
    w_int       ,w_slice        ,a_int          ,a_slice    ,\
    e_int       ,e_slice        ,ao_int         ,ao_slice   ,\
    eo_int      ,eo_slice                                   ,\
    w_ovf       ,w_sp           ,a_ovf          ,a_sp       ,\
    e_ovf       ,e_sp           ,ao_ovf         ,ao_sp      ,\
    eo_ovf      ,eo_sp                                      ,\
    g_int       ,g_slice        ,pw_int         ,pw_slice   ,\
    LDPS_config ,activation    = load_config(config_dict)

    #w_fp_fxp_old = w_fp_fxp
    #w_fp_fxp = w_fp_fxp.replace("_sr", "")
    #on_off_sr(net, a_fp_fxp.replace("_sr", ""),  e_fp_fxp.replace("_sr", ""),
    #               ao_fp_fxp.replace("_sr", ""), eo_fp_fxp.replace("_sr", ""))
    #w_fp_fxp = w_fp_fxp.replace("_dyth", "")
    #on_off_sr(net, a_fp_fxp.replace("_dyth", ""),  e_fp_fxp.replace("_dyth", ""),
    #               ao_fp_fxp.replace("_dyth", ""), eo_fp_fxp.replace("_dyth", ""))
    #on_off_sr(net, a_fp_fxp.replace("_sr", ""),  e_fp_fxp.replace("_sr", ""),
    #               ao_fp_fxp, eo_fp_fxp)
    #on_off_sr(net, a_fp_fxp,  e_fp_fxp,
    #               ao_fp_fxp.replace("_sr", ""), eo_fp_fxp.replace("_sr", ""))
    #on_off_sr(net,  'dfxp_sr_dyth',  'dfxp_sr_dyth',
    #                'dfxp_sr_dyth',  'dfxp_sr_dyth')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #---Weight & Activation & Error Precision Handle---#
            w_int   ,w_ovf  ,w_sp   ,\
            a_int   ,a_ovf  ,a_sp   ,\
            e_int   ,e_ovf  ,e_sp   ,\
            ao_int  ,ao_ovf ,ao_sp  ,\
            eo_int  ,eo_ovf ,eo_sp  = int_adapt(w_int   ,w_ovf      ,w_sp     ,\
                                            a_int       ,a_ovf      ,a_sp     ,\
                                            e_int       ,e_ovf      ,e_sp     ,\
                                            ao_int      ,ao_ovf     ,ao_sp    ,\
                                            eo_int      ,eo_ovf     ,eo_sp    ,\
                                            w_fp_fxp    ,a_fp_fxp   ,e_fp_fxp ,\
                                            ao_fp_fxp   ,eo_fp_fxp)
            
            old_params, old_bias=[], []
            index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    w_ei=w_int[index]
                    w_mf=w_slice[index]-w_int[index] -1
                    old_params.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                    this_is_bias = False
                    index_a = index_a + 1
                elif 'FFO_status' in name:
                    param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                    this_is_bias = False
                    index_ao = index_ao + 1
                elif 'EP_status' in name:
                    param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                    this_is_bias = False
                    index_e = index_e + 1
                elif 'EPO_status' in name:
                    param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                    this_is_bias = False
                    index_eo = index_eo + 1
                elif this_is_bias and not('bw_param' in name):
                    old_bias.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                    this_is_bias = False
                    index_b=index_b+1
                #if 'bw_param' in name :
                #        param.data = torch.tensor(  [ w_int[index-1] , gpu_num])
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            index, index_b = 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    this_is_bias = False
                elif 'FFO_status' in name:
                    this_is_bias = False
                elif 'EP_status' in name:
                    this_is_bias = False
                elif 'EPO_status' in name:
                    this_is_bias = False
                elif this_is_bias and not('bw_param' in name):
                    param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = False
                    index_b=index_b+1
                    
            loss_temp += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_temp/(batch_idx+1), 100.*correct/total, correct, total))
            
            
    #w_fp_fxp = w_fp_fxp_old
    #on_off_sr(net, a_fp_fxp,  e_fp_fxp,
    #               ao_fp_fxp, eo_fp_fxp)

    return 100.*correct/total, loss_temp/(batch_idx+1)

def get_outputs(net, dataloader, device):
    net.eval()
    correct = 0
    total = 0
    output_list = []
    print('..Get Outputs of T-Net..')
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        #if params.cuda:
        #    data_batch, labels_batch = data_batch.cuda(async=True), \
        #                                labels_batch.cuda(async=True)
        '''
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        output_teacher_batch = net(data_batch).data.cpu().numpy()
        output_list.append(output_teacher_batch)
        '''

        
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs).data

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))

        output_list.append(outputs)

        #check_process_mem()
        #check_cuda_mem()

    return output_list

def train_kd(t_net, net, trainloader, device, optimizer, criterion, epoch, g_scale, loss_scale, break_train, config_dict):
    t_net.eval()
    net.train()
    save_data = False
    save_num = 0
    loss_temp = 0
    correct = 0
    total = 0
    loss_list = list()

    gpu_num, pw_fp_fxp, w_fp_fxp, \
    a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp, \
    g_fp_fxp, w_int, w_slice, \
    a_int, a_slice, e_int, e_slice, \
    ao_int, ao_slice, eo_int, eo_slice, \
    w_ovf, w_sp, \
    a_ovf, a_sp, e_ovf, e_sp, \
    ao_ovf, ao_sp, eo_ovf, eo_sp, \
    g_int, g_slice, pw_int, pw_slice, LDPS_config, activation = load_config(config_dict)

    #if (epoch != 0) & (epoch % 2 == 0):
    #    if ('dfxp' in w_fp_fxp):
    #        for list_n in range(1, len(w_slice)-1):
    #            w_slice[list_n] = max(w_slice[list_n] - 2, 2)
    #    if ('dfxp' in a_fp_fxp):
    #        for list_n in range(1, len(a_slice)-1):
    #            a_slice[list_n] = max(a_slice[list_n] - 2, 2)
    #    if ('dfxp' in e_fp_fxp):
    #        for list_n in range(1, len(e_slice)-1):
    #            e_slice[list_n] = max(e_slice[list_n] - 2, 6)
    #    if ('dfxp' in ao_fp_fxp):
    #        for list_n in range(1, len(ao_slice)-1):
    #            ao_slice[list_n] = max(ao_slice[list_n] - 2, 1)
    #    if ('dfxp' in eo_fp_fxp):
    #        for list_n in range(1, len(eo_slice)-1):
    #            eo_slice[list_n] = max(eo_slice[list_n] - 2, 1)
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):  
        #---Weight & Activation & Error Precision Handle---#
        w_int, w_ovf, w_sp, \
        a_int, a_ovf, a_sp, \
        e_int, e_ovf, e_sp, \
        ao_int, ao_ovf, ao_sp, \
        eo_int, eo_ovf, eo_sp = int_adapt(w_int,    w_ovf,    w_sp, \
                                       a_int,    a_ovf,    a_sp, \
                                       e_int,    e_ovf,    e_sp, \
                                       ao_int,   ao_ovf,   ao_sp, \
                                       eo_int,   eo_ovf,   eo_sp, \
                                       w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp)
       
        config_dict = save_config(w_int, w_slice, \
                              a_int, a_slice, e_int, e_slice, \
                              ao_int, ao_slice, eo_int, eo_slice, \
                              w_ovf, w_sp, \
                              a_ovf, a_sp, e_ovf, e_sp, \
                              ao_ovf, ao_sp, eo_ovf, eo_sp, \
                              config_dict)


        
        w_change, a_change, e_change, bw_change, is_LDPS = LDPS_control_1(batch_idx, w_fp_fxp[:4], LDPS_config)
        config_dict['bw_change'] = bw_change
        
        old_params, old_bias=[], []
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        this_is_bias = False
        for name, param in net.named_parameters():
            if param.dim() != 1:
                w_ei=w_int[index]
                w_mf=w_slice[index]-w_int[index] -1
                w_bw = w_slice[index]
                old_params.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                w_ovf[index], w_sp[index] = ovf_sp(old_params[-1], param.data,w_bw, w_ei, w_ovf[index], w_sp[index], w_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                old_bias.append(param.data)
                param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                w_ovf[index-1], w_sp[index-1] = ovf_sp(old_bias[-1], param.data,w_bw,  w_ei, w_ovf[index-1], w_sp[index-1], w_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
            if 'bw_param' in name :
                param.data = torch.tensor( [ w_int[index-1] , gpu_num])
                #print(name)

        #---Feed Forward---#
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        t_results = t_net(inputs).data
        outputs = net(inputs)

        #---Error Propagation---#
        #loss = loss_scale* loss_fn_kd(outputs, targets, t_results[batch_idx].to(device))
        loss = loss_scale* loss_fn_kd(outputs, targets, t_results)
        #loss = loss_scale*criterion(outputs, targets)
        loss.backward()
        
        #---Gradient Precision Handle---#
        #---Overflow & Surplus Detection + Primal Weight Recovery---#
        index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
        for name, param in net.named_parameters():
            if param.dim() != 1:
                if ('dfxp' in g_fp_fxp) :
                    param.grad=custom_precision(param.grad, w_int[index]+g_scale, g_slice-w_int[index]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                #save_mat(save_data & (batch_idx >= save_num), param.grad.cpu().numpy(), 'grad', index)
                param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index]+1, pw_slice-w_int[index]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = True
                index=index+1
            elif 'FF_status' in name:
                a_ovf[index_a] = float(param.data[2])
                a_sp[index_a] = float(param.data[3])
                this_is_bias = False
                index_a = index_a + 1
            elif 'FFO_status' in name:
                ao_ovf[index_ao] = float(param.data[2])
                ao_sp[index_ao] = float(param.data[3])
                this_is_bias = False
                index_ao = index_ao + 1
            elif 'EP_status' in name:
                e_ovf[index_e] = float(param.data[2])
                e_sp[index_e] = float(param.data[3])
                this_is_bias = False
                index_e = index_e + 1
            elif 'EPO_status' in name:
                eo_ovf[index_eo] = float(param.data[2])
                eo_sp[index_eo] = float(param.data[3])
                this_is_bias = False
                index_eo = index_eo + 1
            elif this_is_bias and not('bw_param' in name):
                if ('dfxp' in g_fp_fxp):
                    param.grad=custom_precision(param.grad, w_int[index-1] + g_scale, g_slice-w_int[index-1]-g_scale-1, g_fp_fxp)
                else:
                    param.grad=custom_precision(param.grad, g_int, g_slice-g_int-1, g_fp_fxp)
                param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                if ('dfxp' in pw_fp_fxp):
                    param.data=custom_precision(param.data, w_int[index-1] + 1, pw_slice-w_int[index-1]-2, pw_fp_fxp)
                else:
                    param.data=custom_precision(param.data, pw_int, pw_slice-pw_int-1, pw_fp_fxp)
                this_is_bias = False
                index_b=index_b+1
                

        if (break_train):
            loss_temp = 0
            total = 1
            correct = 0
            break
        else:
            #---Optimizer (w Update)---#
            optimizer.step()
            loss_temp += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_temp/(batch_idx+1), 100.*correct/total, correct, total))
    

    config_dict = save_config(w_int, w_slice, \
                              a_int, a_slice, e_int, e_slice, \
                              ao_int, ao_slice, eo_int, eo_slice, \
                              w_ovf, w_sp, \
                              a_ovf, a_sp, e_ovf, e_sp, \
                              ao_ovf, ao_sp, eo_ovf, eo_sp, \
                              config_dict)

    return 100.*correct/total, loss_temp/(batch_idx+1), config_dict , loss_list


def test_nosr(net, testloader, device, optimizer, criterion, epoch, config_dict):
    net.eval()
    loss_temp = 0
    correct = 0
    total = 0

    gpu_num     ,pw_fp_fxp      ,w_fp_fxp       ,a_fp_fxp   ,\
    e_fp_fxp    ,ao_fp_fxp      ,eo_fp_fxp      ,g_fp_fxp   ,\
    w_int       ,w_slice        ,a_int          ,a_slice    ,\
    e_int       ,e_slice        ,ao_int         ,ao_slice   ,\
    eo_int      ,eo_slice                                   ,\
    w_ovf       ,w_sp           ,a_ovf          ,a_sp       ,\
    e_ovf       ,e_sp           ,ao_ovf         ,ao_sp      ,\
    eo_ovf      ,eo_sp                                      ,\
    g_int       ,g_slice        ,pw_int         ,pw_slice   ,\
    LDPS_config ,activation    = load_config(config_dict)

    w_fp_fxp_old = w_fp_fxp
    w_fp_fxp = w_fp_fxp.replace("_sr", "")
    on_off_sr(net, a_fp_fxp.replace("_sr", ""),  e_fp_fxp.replace("_sr", ""),
                   ao_fp_fxp.replace("_sr", ""), eo_fp_fxp.replace("_sr", ""))
    #w_fp_fxp = w_fp_fxp.replace("_dyth", "")
    #on_off_sr(net, a_fp_fxp.replace("_dyth", ""),  e_fp_fxp.replace("_dyth", ""),
    #               ao_fp_fxp.replace("_dyth", ""), eo_fp_fxp.replace("_dyth", ""))
    #on_off_sr(net, a_fp_fxp.replace("_sr", ""),  e_fp_fxp.replace("_sr", ""),
    #               ao_fp_fxp, eo_fp_fxp)
    #on_off_sr(net, a_fp_fxp,  e_fp_fxp,
    #               ao_fp_fxp.replace("_sr", ""), eo_fp_fxp.replace("_sr", ""))
    #w_fp_fxp = 'dfxp_sr_dyth'
    #on_off_sr(net,  'dfxp_sr_dyth',  'dfxp_sr_dyth',
    #                'dfxp_sr_dyth',  'dfxp_sr_dyth')

    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #---Weight & Activation & Error Precision Handle---#
            w_int   ,w_ovf  ,w_sp   ,\
            a_int   ,a_ovf  ,a_sp   ,\
            e_int   ,e_ovf  ,e_sp   ,\
            ao_int  ,ao_ovf ,ao_sp  ,\
            eo_int  ,eo_ovf ,eo_sp  = int_adapt(w_int   ,w_ovf      ,w_sp     ,\
                                            a_int       ,a_ovf      ,a_sp     ,\
                                            e_int       ,e_ovf      ,e_sp     ,\
                                            ao_int      ,ao_ovf     ,ao_sp    ,\
                                            eo_int      ,eo_ovf     ,eo_sp    ,\
                                            w_fp_fxp    ,a_fp_fxp   ,e_fp_fxp ,\
                                            ao_fp_fxp   ,eo_fp_fxp)
            
            old_params, old_bias=[], []
            index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    w_ei=w_int[index]
                    w_mf=w_slice[index]-w_int[index] -1
                    old_params.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                    this_is_bias = False
                    index_a = index_a + 1
                elif 'FFO_status' in name:
                    param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                    this_is_bias = False
                    index_ao = index_ao + 1
                elif 'EP_status' in name:
                    param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                    this_is_bias = False
                    index_e = index_e + 1
                elif 'EPO_status' in name:
                    param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                    this_is_bias = False
                    index_eo = index_eo + 1
                elif this_is_bias and not('bw_param' in name):
                    old_bias.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                    this_is_bias = False
                    index_b=index_b+1
                #if 'bw_param' in name :
                #        param.data = torch.tensor(  [ w_int[index-1] , gpu_num])
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            index, index_b = 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    this_is_bias = False
                elif 'FFO_status' in name:
                    this_is_bias = False
                elif 'EP_status' in name:
                    this_is_bias = False
                elif 'EPO_status' in name:
                    this_is_bias = False
                elif this_is_bias and not('bw_param' in name):
                    param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = False
                    index_b=index_b+1
                    
            loss_temp += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_temp/(batch_idx+1), 100.*correct/total, correct, total))
            
            
    w_fp_fxp = w_fp_fxp_old
    on_off_sr(net, a_fp_fxp,  e_fp_fxp,
                   ao_fp_fxp, eo_fp_fxp)

    return 100.*correct/total, loss_temp/(batch_idx+1)

def test_sr(net, testloader, device, optimizer, criterion, epoch, config_dict):
    net.eval()
    loss_temp = 0
    correct = 0
    total = 0

    gpu_num     ,pw_fp_fxp      ,w_fp_fxp       ,a_fp_fxp   ,\
    e_fp_fxp    ,ao_fp_fxp      ,eo_fp_fxp      ,g_fp_fxp   ,\
    w_int       ,w_slice        ,a_int          ,a_slice    ,\
    e_int       ,e_slice        ,ao_int         ,ao_slice   ,\
    eo_int      ,eo_slice                                   ,\
    w_ovf       ,w_sp           ,a_ovf          ,a_sp       ,\
    e_ovf       ,e_sp           ,ao_ovf         ,ao_sp      ,\
    eo_ovf      ,eo_sp                                      ,\
    g_int       ,g_slice        ,pw_int         ,pw_slice   ,\
    LDPS_config ,activation    = load_config(config_dict)

    w_fp_fxp_old = w_fp_fxp
    #w_fp_fxp = w_fp_fxp.replace("_sr", "")
    #on_off_sr(net, a_fp_fxp.replace("_sr", ""),  e_fp_fxp.replace("_sr", ""),
    #               ao_fp_fxp.replace("_sr", ""), eo_fp_fxp.replace("_sr", ""))
    #w_fp_fxp = w_fp_fxp.replace("_dyth", "")
    #on_off_sr(net, a_fp_fxp.replace("_dyth", ""),  e_fp_fxp.replace("_dyth", ""),
    #               ao_fp_fxp.replace("_dyth", ""), eo_fp_fxp.replace("_dyth", ""))
    #on_off_sr(net, a_fp_fxp.replace("_sr", ""),  e_fp_fxp.replace("_sr", ""),
    #               ao_fp_fxp, eo_fp_fxp)
    #on_off_sr(net, a_fp_fxp,  e_fp_fxp,
    #               ao_fp_fxp.replace("_sr", ""), eo_fp_fxp.replace("_sr", ""))
    w_fp_fxp = 'dfxp_sr_dyth'
    on_off_sr(net,  'dfxp_sr_dyth',  'dfxp_sr_dyth',
                    'dfxp_sr_dyth',  'dfxp_sr_dyth')

    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #---Weight & Activation & Error Precision Handle---#
            w_int   ,w_ovf  ,w_sp   ,\
            a_int   ,a_ovf  ,a_sp   ,\
            e_int   ,e_ovf  ,e_sp   ,\
            ao_int  ,ao_ovf ,ao_sp  ,\
            eo_int  ,eo_ovf ,eo_sp  = int_adapt(w_int   ,w_ovf      ,w_sp     ,\
                                            a_int       ,a_ovf      ,a_sp     ,\
                                            e_int       ,e_ovf      ,e_sp     ,\
                                            ao_int      ,ao_ovf     ,ao_sp    ,\
                                            eo_int      ,eo_ovf     ,eo_sp    ,\
                                            w_fp_fxp    ,a_fp_fxp   ,e_fp_fxp ,\
                                            ao_fp_fxp   ,eo_fp_fxp)
            
            old_params, old_bias=[], []
            index, index_b, index_a, index_e, index_ao, index_eo = 0, 0, 0, 0, 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    w_ei=w_int[index]
                    w_mf=w_slice[index]-w_int[index] -1
                    old_params.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf, w_fp_fxp)
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    param.data=is2if_transform(a_int[index_a], a_slice[index_a])
                    this_is_bias = False
                    index_a = index_a + 1
                elif 'FFO_status' in name:
                    param.data=is2if_transform(ao_int[index_ao], ao_slice[index_ao])
                    this_is_bias = False
                    index_ao = index_ao + 1
                elif 'EP_status' in name:
                    param.data=is2if_transform(e_int[index_e], e_slice[index_e])
                    this_is_bias = False
                    index_e = index_e + 1
                elif 'EPO_status' in name:
                    param.data=is2if_transform(eo_int[index_eo], eo_slice[index_eo])
                    this_is_bias = False
                    index_eo = index_eo + 1
                elif this_is_bias and not('bw_param' in name):
                    old_bias.append(param.data)
                    param.data=custom_precision(param.data, w_ei, w_mf + a_slice[index-1], w_fp_fxp)
                    this_is_bias = False
                    index_b=index_b+1
                #if 'bw_param' in name :
                #        param.data = torch.tensor(  [ w_int[index-1] , gpu_num])
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            index, index_b = 0, 0
            this_is_bias = False
            for name, param in net.named_parameters():
                if param.dim() != 1:
                    param.data=old_params[index].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = True
                    index=index+1
                elif 'FF_status' in name:
                    this_is_bias = False
                elif 'FFO_status' in name:
                    this_is_bias = False
                elif 'EP_status' in name:
                    this_is_bias = False
                elif 'EPO_status' in name:
                    this_is_bias = False
                elif this_is_bias and not('bw_param' in name):
                    param.data=old_bias[index_b].float().to(torch.device('cuda:'+str(gpu_num)))
                    this_is_bias = False
                    index_b=index_b+1
                    
            loss_temp += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_temp/(batch_idx+1), 100.*correct/total, correct, total))
            
            
    w_fp_fxp = w_fp_fxp_old
    on_off_sr(net, a_fp_fxp,  e_fp_fxp,
                   ao_fp_fxp, eo_fp_fxp)

    return 100.*correct/total, loss_temp/(batch_idx+1)



