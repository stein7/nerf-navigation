import torch
import pdb
import lib.gpu_alloc as ga

def check_int(w_ovf, w_sp, 
              a_ovf, a_sp, 
              e_ovf, e_sp, 
              ao_ovf, ao_sp, 
              eo_ovf, eo_sp,
              w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp):

    w_check = True
    a_check = True
    e_check = True
    ao_check = True
    eo_check = True

    if ('dfxp' in w_fp_fxp):
        for list_n in range(0, len(w_ovf)):
            if w_ovf[list_n] != 0.:
                w_check = False
            if w_sp[list_n] != 0.:
                w_check = False
                
    if ('dfxp' in a_fp_fxp):
        for list_n in range(0, len(a_ovf)):
            if a_ovf[list_n] != 0.:
                a_check = False
            if a_sp[list_n] != 0.:
                a_check = False

    if ('dfxp' in e_fp_fxp):
        for list_n in range(0, len(e_ovf)):
            if e_ovf[list_n] != 0.:
                e_check = False
            if e_sp[list_n] != 0.:
                e_check = False
                
    if ('dfxp' in ao_fp_fxp):
        for list_n in range(0, len(ao_ovf)):
            if ao_ovf[list_n] != 0.:
                ao_check = False
            if ao_sp[list_n] != 0.:
                ao_check = False

    if ('dfxp' in eo_fp_fxp):
        for list_n in range(0, len(eo_ovf)):
            if eo_ovf[list_n] != 0.:
                eo_check = False
            if eo_sp[list_n] != 0.:
                eo_check = False

    return w_check, a_check, e_check, ao_check, eo_check

def int_adapt(w_int, w_ovf, w_sp, 
              a_int, a_ovf, a_sp, 
              e_int, e_ovf, e_sp, 
              ao_int, ao_ovf, ao_sp, 
              eo_int, eo_ovf, eo_sp,
              w_fp_fxp, a_fp_fxp, e_fp_fxp, ao_fp_fxp, eo_fp_fxp, ):

    if ('dfxp' in w_fp_fxp):
        for list_n in range(0, len(w_ovf)):
            if w_ovf[list_n] != 0.:
                w_int[list_n] = w_int[list_n] + 1
                w_ovf[list_n] = 0.
            elif w_sp[list_n] != 0.:
                w_int[list_n] = w_int[list_n] - 1
                w_sp[list_n] = 0.
    if ('dfxp' in a_fp_fxp):
        for list_n in range(0, len(a_ovf)):
            if a_ovf[list_n] != 0.:
                a_int[list_n] = a_int[list_n] + 1
                a_ovf[list_n] = 0.
            elif a_sp[list_n] != 0.:
                a_int[list_n] = a_int[list_n] - 1
                a_sp[list_n] = 0.
    if ('dfxp' in e_fp_fxp):
        for list_n in range(0, len(e_ovf)):
            if e_ovf[list_n] != 0.:
                e_int[list_n] = e_int[list_n] + 1
                e_ovf[list_n] = 0.
            elif e_sp[list_n] != 0.:
                e_int[list_n] = e_int[list_n] - 1
                e_sp[list_n] = 0.
    if ('dfxp' in ao_fp_fxp):
        for list_n in range(0, len(ao_ovf)):
            if ao_ovf[list_n] != 0.:
                ao_int[list_n] = ao_int[list_n] + 1
                ao_ovf[list_n] = 0.
            elif ao_sp[list_n] != 0.:
                ao_int[list_n] = ao_int[list_n] - 1
                ao_sp[list_n] = 0.
    if ('dfxp' in eo_fp_fxp):
        for list_n in range(0, len(eo_ovf)):
            if eo_ovf[list_n] != 0.:
                eo_int[list_n] = eo_int[list_n] + 1
                eo_ovf[list_n] = 0.
            elif eo_sp[list_n] != 0.:
                eo_int[list_n] = eo_int[list_n] - 1
                eo_sp[list_n] = 0.

    return w_int, w_ovf, w_sp, a_int, a_ovf, a_sp, e_int, e_ovf, e_sp, ao_int, ao_ovf, ao_sp, eo_int, eo_ovf, eo_sp

def is2if_transform(i, s):
    return torch.tensor([i, s-i-1, 0., 0.], device=ga.device_gpu)

def ovf_sp(origin, out, bw, e_i, ovf, sp, fp_fxp):
    if ('dfxp' in fp_fxp):
        ovf_max = torch.pow(torch.tensor(2.0, device=ga.device_gpu), e_i)
        sp_max = torch.pow(torch.tensor(2.0, device=ga.device_gpu), e_i-1)
        #print(( (fp_fxp[-5:] == '_dyth') and ( bw <= 6 )  ) or (fp_fxp[-3:] == '_th' ))
        if ( (fp_fxp[-5:] == '_dyth') and ( bw <= 5 )  ) or (fp_fxp[-3:] == '_th' ):
            #print('threshold')
            ovf_th = ga.sr_thresh*torch.rand(1)
            #ovf_th = 0.0001
            for i in range(len(origin.size())):
                ovf_th = ovf_th * origin.size()[i]
        else:
            ovf_th = 0

        ovf = ovf + (len(origin[torch.abs(origin) > ovf_max]) > ovf_th)
        sp = sp + (len(origin[torch.abs(origin) > sp_max]) <= ovf_th)
    else:
        ovf = 0
        sp = 0
    return ovf, sp

    
def EP_ovf_sp_detection(self, grad_input, grad_output):
    e_i = self.EP_status[0]
    bw = self.EP_status[0] + self.EP_status[1] +1  
    ovf , sp = ovf_sp(grad_output[0], grad_input, bw, e_i, 0, 0, self.EP_fp_fxp)
    
    self.EP_status[2] = ovf
    self.EP_status[3] = sp

def EP_ovf_sp_detection2(self, grad_input, grad_output):
    e_i = self.FFO_status[0]
    bw = self.FFO_status[0] + self.FFO_status[1] +1
    ovf , sp = ovf_sp(grad_output[0], grad_input, bw, e_i, 0, 0, self.FFO_fp_fxp)
    
    self.FFO_status[2] = ovf
    self.FFO_status[3] = sp

def add_EP_backward_hook(net, net_config, rep_num):

    index_f, index_e = 0, 0
    for n1 in net._modules.keys():
        #if hasattr(net._modules[n1], 'FF'):
        if 'Change_precision_FF()' == str(net._modules[n1]):
            net._modules[n1].register_backward_hook(EP_ovf_sp_detection2)
            index_f += 1
        elif 'Change_precision_EP()' == str(net._modules[n1]):
            net._modules[n1].register_backward_hook(EP_ovf_sp_detection)
            index_e += 1

        if 'OrderedDict()' in str(net._modules[n1]._modules):
            continue
        else:
            for n2 in net._modules[n1]._modules.keys():
                #print(net._modules[n1]._modules[n2])
                if 'Change_precision_FF()' == str(net._modules[n1]._modules[n2]):
                    net._modules[n1]._modules[n2].register_backward_hook(EP_ovf_sp_detection2)
                    index_f += 1
                elif 'Change_precision_EP()' == str(net._modules[n1]._modules[n2]):
                    net._modules[n1]._modules[n2].register_backward_hook(EP_ovf_sp_detection)
                    index_e += 1

                if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules):
                    continue
                else:
                    for n3 in net._modules[n1]._modules[n2]._modules.keys():
                        #print(net._modules[n1]._modules[n2]._modules[n3])
                        if 'Change_precision_FF()' == str(net._modules[n1]._modules[n2]._modules[n3]):
                            net._modules[n1]._modules[n2]._modules[n3].register_backward_hook(EP_ovf_sp_detection2)
                            index_f += 1
                        elif 'Change_precision_EP()' == str(net._modules[n1]._modules[n2]._modules[n3]):
                            net._modules[n1]._modules[n2]._modules[n3].register_backward_hook(EP_ovf_sp_detection)
                            index_e += 1
                        if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules[n3]._modules):
                            continue
                        else:
                            for n4 in net._modules[n1]._modules[n2]._modules[n3]._modules.keys():
                                #print(net._modules[n1]._modules[n2]._modules[n3]._modules[n4])
                                if 'Change_precision_FF()' == str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]):
                                    net._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_backward_hook(EP_ovf_sp_detection2)
                                    index_f += 1
                                elif 'Change_precision_EP()' == str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]):
                                    net._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_backward_hook(EP_ovf_sp_detection)
                                    index_e += 1
                                if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules):
                                    continue
                                else:
                                    for n5 in net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules.keys():
                                        #print(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5])
                                        if 'Change_precision_FF()' == str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]):
                                            net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_backward_hook(EP_ovf_sp_detection2)
                                            index_f += 1
                                        elif 'Change_precision_EP()' == str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]):
                                            net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_backward_hook(EP_ovf_sp_detection)
                                            index_e += 1
                                        if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]._modules):
                                            continue
                                        else:
                                            pdb.set_trace()
    print(index_f)
    print(index_e)
    if ((index_f == index_e) & (index_f == rep_num)):
        print("Successfully Adding Backward Hook!")
    else:
        print("Error: Wrong Backward Hook")

#===================== Backward Weight Observing ==================
def add_afhook(net):
    index = 0 
    for n1 in net._modules.keys():
        if ( 'AsymmetricFeedbackConv2d' in str(net._modules[n1])[:24] )   and not ('Sequential' in str(net._modules[n1]) ) :
            net._modules[n1].register_backward_hook(BW_detect)
            index += 1
        if 'OrderedDict()' in str(net._modules[n1]._modules):
            continue
        else :
            for n2 in net._modules[n1]._modules.keys():
                #print(net._modules[n1]._modules[n2])
                if ('AsymmetricFeedbackConv2d' in str(net._modules[n1]._modules[n2])[:24] )   and not ('Sequential' in str(net._modules[n1]._modules[n2]) ) :                
                    net._modules[n1]._modules[n2].register_backward_hook(BW_detect)
                    index += 1 

                if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules):
                    continue
                else :
                    for n3 in net._modules[n1]._modules[n2]._modules.keys():
                        if ('AsymmetricFeedbackConv2d' in str(net._modules[n1]._modules[n2]._modules[n3])[:24])    and not ('Sequential' in str(net._modules[n1]._modules[n2]._modules[n3]) ) :                            
                            net._modules[n1]._modules[n2]._modules[n3].register_backward_hook(BW_detect)
                            index += 1

                        if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules[n3]._modules):
                            continue
                        else:
                            for n4 in net._modules[n1]._modules[n2]._modules[n3]._modules.keys() :
                                if ('AsymmetricFeedbackConv2d' in str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4])[:24])  and not ('Sequential' in str(net._modules[n1]._modules[n2]._moduels[n3]._modules[n4]) )  :                            
                                    net._modules[n1]._modules[n2]._modules[n3]._modules[n4].register_backward_hook(BW_detect)
                                    index += 1
                                if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules):
                                    continue
                                else:
                                    for n5 in net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules.keys():                                
                                        if ('AsymmetricFeedbackConv2d' in str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5])[:24])   and not ('Sequential' in str(net._modules[n1]._modules[n2]._moduels[n3]._modules[n4]._modules[n5]) ) :                        
                                            net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5].register_backward_hook(BW_detect)
                                            index += 1 
                                        if 'OrderedDict()' in str(net._modules[n1]._modules[n2]._modules[n3]._modules[n4]._modules[n5]._modules):
                                            continue
                                        else:
                                            pdb.set_trace() 

    print('Add BW Detect hook   :  ',  index )

'''
def add_AF_hook(net_lower_class):
    for k, layer in enumerate(net_lower_class):
        name = layer.__class__.__name__
        #print(name)
        if 'Conv' in name:
            layer.register_backward_hook(BW_detect)'''

def BW_detect(self, grad_input, grad_output):
    #try : 
    #print('BW_config  : ', self.bw_bw, self.bw_param[0])
    #print('BW         : ', self.feedback_weight )
    #print('FW         : ', self.weight.dim())
    try:
        pass
        #print(self)
        #print('FW',  self.weight[0][0][0])
        #print('BW', self.feedback_weight[0][0][0])
        #print('GR_IN      : ', grad_input[0][0][0])
        #print('mean : ', torch.mean(grad_input[0][0][0]), 'std  : ', torch.std(grad_input[0][0][0]))
        #print('GR_OUT     ; ', grad_output[0][0][0])
    except : pass
    
    #except : pass

#=======================================================================
