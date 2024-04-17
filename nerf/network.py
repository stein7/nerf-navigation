import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

import pdb
from lib.precision import *
import math

import csv
import numpy as np

from torch.autograd import Function

from tqdm import tqdm

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt=None,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 fxp=False,
                 fxp_bw=8,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)
        
        self.opt = opt
        self.fxp = self.opt.w_fxp
        self.fxp_bw = self.opt.w_fxp_bw
        self.encoding = encoding

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        
        
        self.bg_size = 32
        sigma_net = []
        sigma_pruned_net = []
        sigma_quant_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            sigma_pruned_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if self.opt.compressor_accuracy:
                sigma_quant_net.append(AQFC(in_dim, out_dim, bias=False, w_fxp_bw=self.opt.w_fxp_bw, i_fxp = self.opt.i_fxp, i_fxp_bw=self.opt.i_fxp_bw, err_fxp=self.opt.err_fxp, err_fxp_bw=self.opt.err_fxp_bw, l=l, at_fan_in=self.opt.at_fan_in, compressor_accuracy=self.opt.compressor_accuracy))
            else:
                sigma_quant_net.append(QFC(in_dim, out_dim, bias=False, i_fxp = self.opt.i_fxp, i_fxp_bw=self.opt.i_fxp_bw, err_fxp=self.opt.err_fxp, err_fxp_bw=self.opt.err_fxp_bw, l=l))
        

        self.sigma_net = nn.ModuleList(sigma_net)
        self.sigma_pruned_net = nn.ModuleList(sigma_pruned_net)
        self.sigma_quant_net = nn.ModuleList(sigma_quant_net)

        self.sigma_net_prun_th_ratio = [0,0,0.5,0.5,0.5,0]


        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        del x
        torch.cuda.empty_cache()
        #pdb.set_trace()
        for l in range(self.num_layers):
            #print("forward")
            #### SR Quantization ####                
            h = self.sigma_net[l](h)
            #h = self.sigma_prun_net[l](h)
            #if self.opt.err_fxp:
            #    with torch.no_grad():
            #        h = self.sigma_quant_net[l](h)
            #else:
            #    h = self.sigma_quant_net[l](h)
            
            #pdb.set_trace()
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

            if self.opt.i_fxp and l<self.num_layers-1:            
                i_int_bw = int(math.log2(h.max()))
                h = custom_precision(h, i_int_bw, self.opt.i_fxp_bw-1-i_int_bw, 'fxp')
            #pdb.set_trace()
        
        #pdb.set_trace()
        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        del h
        torch.cuda.empty_cache()

        # color 
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color
    
    def quant_density(self, x):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        if self.encoding == 'frequency':
            x = self.encoder(x)
        elif self.encoding == 'hashgrid':
            x = self.encoder(x, bound=self.bound)
        elif self.encoding == 'tiledgrid':
            x = self.encoder(x)

        h = x
        #pdb.set_trace()
        for l in range(self.num_layers):
            #### SR Quantization ####                
            h = self.sigma_quant_net[l](h)

            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        
        sigma = trunc_exp(h[..., 0])


        return sigma

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        #print(self.sigma_net)
        #print("nerwork.py (151) density func")
        if self.encoding == 'frequency':
            x = self.encoder(x)
        elif self.encoding == 'hashgrid':
            x = self.encoder(x, bound=self.bound)
        elif self.encoding == 'tiledgrid':
            x = self.encoder(x)
        
        h = x

        if not(self.opt.train):
            h_prun = x
            h_quant = x

        h_batch, h_ich = h.shape

        for l in range(self.num_layers):
            print(l,"th layer inference")
            print(l,"th layer batch num : ",h_batch)

            inputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_origin_input.csv"
            outputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_origin_output.csv"
            prun_inputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_prun_input.csv"
            prun_outputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_prun_output.csv"

            #### SR Quantization ####
            #if self.fxp and l>0 and l<self.num_layers-1:
            #if self.opt.i_fxp and l>0:
            #    i_int_bw = int(math.log2(h_quant.max()))
            #    h_quant = custom_precision(h_quant, i_int_bw, self.opt.i_fxp_bw-1-i_int_bw, 'fxp')
            #if self.fxp and l>0 and l<self.num_layers-1:
            #if self.opt.i_fxp and l>0:
            #    i_int_bw = int(math.log2(h_prun.max()))
            #    h_prun = custom_precision(h_prun, i_int_bw, self.opt.i_fxp_bw-1-i_int_bw, 'fxp')

            #### SR input data log ####
            #with open (inputfile_name, 'w', newline='') as f:
            #    writer = csv.writer(f)
            #    writer.writerows(h.cpu().numpy())
            #    torch.save(h,"./intermediate/sigmaNet_layer" + str(l) + "_origin_input.pth")
            #with open (prun_inputfile_name, 'w', newline='') as f:
            #    writer = csv.writer(f)
            #    writer.writerows(h_prun.cpu().numpy())
            #    torch.save(h_prun,"./intermediate/sigmaNet_layer" + str(l) + "_prun_input.pth")
            
            #### Inference & Activation Func ####
            h_batch, h_ich = h.shape


            #### Inference using Batch Group ####

            ### 1st layer Inference ###
            if l==0 :
                
                if self.opt.approx:
                    w_cur_layer = self.sigma_quant_net[l].weight
    
                    weight = w_cur_layer.T
                    pdb.set_trace()
    
    
                    ### Check ich prun ###
                    weight_ich_prun = torch.sum(weight==0, dim=1)                               # row directional = ich directional
                    weight_remain_ich = weight_ich_prun != weight.shape[1]
                    weight = weight[weight_remain_ich]                                          # check coarse weight pruning
    
                    weight_not_zero = weight[weight!=0]
                    weight_smallest_unit = torch.min(torch.abs(weight_not_zero))
                    weight_q = weight/weight_smallest_unit
                    weight_bin = dec2bin(weight_q.int(), self.opt.w_fxp_bw)

                    ### SR Approx Accuracy Sim

                    weight_inv = torch.logical_not(weight_bin)
                    weight_zero_bit_cnt = torch.sum(weight_inv, dim=2)
                    weight_zero_bit_ich = torch.sum(weight_zero_bit_cnt, dim=1)
    
                    weight_zero_bit_cnt_sort = weight_zero_bit_cnt.clone().detach()
                    weight_bin_dup = weight_bin.clone().detach()
                    
                    sort_val, sort_idx = torch.sort(weight_zero_bit_ich, descending=True)
                    top_4k = sort_idx[0:round(sort_idx.shape[0]/4)*2]
    
                    top_4k_slice = top_4k[0:round(sort_idx.shape[0]/4)]
                    zero_max_list = weight_zero_bit_ich[top_4k_slice]
    
    
    
    
                    for ich in range (0, weight_zero_bit_ich.shape[0], 4):
                        weight_zero_bit_cnt_sort[ich,:] = weight_zero_bit_cnt[top_4k_slice[int(ich/4)],:]
    
                        similarity = torch.logical_and(weight_bin[top_4k_slice[int(ich/4)]], weight_bin_dup)
    
                        match_idx = torch.argmin(torch.sum(torch.sum(similarity, dim=2), dim=1))
    
                        weight_zero_bit_cnt_sort[ich+1,:] = weight_zero_bit_cnt[match_idx,:]
    
    
                    for och in range (0, weight_zero_bit_cnt.shape[1], 32):
    
                        weight_zero_bit_cnt_och32 = torch.sum(weight_zero_bit_cnt_sort[:,och:och+32], dim=1)
                        
                        if och == 0:
                            weight_zero_bit_cnt_sort_och32 = weight_zero_bit_cnt_och32.reshape(-1,1)
    
                        else:
                            weight_zero_bit_cnt_sort_och32 = torch.cat([weight_zero_bit_cnt_sort_och32, weight_zero_bit_cnt_och32.reshape(-1,1)], dim=1)
    
    
    
    
                    weight_loop = round(weight.shape[0]/4)
    
    
                    
                    for loop in range (weight_loop):
                        if loop == 0:
                            non_shuffled_zero = weight_zero_bit_ich[0].reshape(1)
                        else:
                            non_shuffled_zero = torch.cat([non_shuffled_zero, weight_zero_bit_ich[loop*4].reshape(1)], dim=0)
    
                    for ich_och32 in range(0, weight_zero_bit_cnt_sort_och32.shape[0], 4):
                        ich_zero_cnt = 0
    
                        for och_och32 in range (weight_zero_bit_cnt_sort_och32.shape[1]):
                            max_element = torch.max(weight_zero_bit_cnt_sort_och32[ich_och32:ich_och32+4, och_och32])
                            ich_zero_cnt = ich_zero_cnt + max_element
    
                        if ich_och32 == 0:
                            och32_shuffled_list = ich_zero_cnt.reshape(1)
                        else:
                            och32_shuffled_list = torch.cat([och32_shuffled_list, ich_zero_cnt.reshape(1)], dim=0)
                            
    
                    print(l,"th layer zero_max idx (before shuffle):            ", non_shuffled_zero, " abs: ", torch.sum(non_shuffled_zero))
                    print(l,"th layer zero_max idx (after och 128 shuffle):     ", zero_max_list, " abs: ", torch.sum(zero_max_list))
                    print(l,"th layer zero_max idx (after och 32 shuffle):      ", och32_shuffled_list, " abs: ", torch.sum(och32_shuffled_list))
                    print(l,"th layer zero_max idx och 128 shuffle performance: ", torch.sum(zero_max_list-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                    print(l,"th layer zero_max idx och 32 shuffle performance:  ", torch.sum(och32_shuffled_list-non_shuffled_zero)/torch.sum(non_shuffled_zero))
    
                    #### Input Reordering ####
                    h_prun_reorder = h_prun[:, top_4k_slice]
                    h_prun_reorder_not_zero = h_prun_reorder[h_prun_reorder != 0] 
                    input_smallest_unit = torch.min(torch.abs(h_prun_reorder_not_zero))
                    input_q = h_prun_reorder/input_smallest_unit
                    h_prun_reorder_bin = dec2bin(input_q.int(), self.opt.i_fxp_bw)
                    h_prun_reorder_bin_inv = torch.logical_not(h_prun_reorder_bin)
                    h_prun_reorder_bin_zero_bit_cnt = torch.sum(h_prun_reorder_bin_inv, dim=2)
    
    
                    h_prun_top4k_sum = torch.round(128*torch.sum(h_prun_reorder_bin_zero_bit_cnt[0:256, :], dim=0)/256).int()
    
                    final = torch.max(och32_shuffled_list,h_prun_top4k_sum)
    
                    print(l,"th layer och32 input feeding                    :  ", h_prun_top4k_sum, " abs: ", torch.sum(h_prun_top4k_sum))
                    print(l,"th layer och32 shuffle+input feeding            :  ", final, " abs: ", torch.sum(final))
                    print(l,"th layer och32 shuffle+input feeding performance:  ", torch.sum(final-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                    print("Batch = ", h_prun_reorder.shape[0], ", ICH = ", h_prun_reorder.shape[1], ", OCH = ", 128, ", Total Weight Zero Bit = ", h_prun_reorder.shape[1]*128*4)
                    pdb.set_trace()
                
                #### Inference ####
                h = self.sigma_net[l](h)
                if not(self.opt.train):
                    h_prun = self.sigma_pruned_net[l](h_prun)
                    with torch.no_grad():
                        h_quant = self.sigma_quant_net[l](h_quant)
                
                if l != self.num_layers-1:
                    h = F.relu(h, inplace=True)
                    if not(self.opt.train):
                        h_prun = F.relu(h_prun, inplace=True)
                        h_quant = F.relu(h_quant, inplace=True)
                
                if self.opt.i_fxp:
                    i_int_bw = int(math.log2(h_prun.max()))
                    h_prun = custom_precision( h_prun, i_int_bw, self.opt.i_fxp_bw-1-i_int_bw, 'fxp')
            
            ### >1st layer Inference ###
            else:

                # Original
                save_h = h
                h = self.sigma_net[l](h)
                if not(self.opt.train):
                    with torch.no_grad():
                        h_quant = self.sigma_quant_net[l](h_quant)
                if l != self.num_layers-1:
                    if not(self.opt.train):
                        h = F.relu(h, inplace=True)
                        h_quant = F.relu(h_quant, inplace=True)
                
                # Pruning using batch group
                bg_num = int(h_batch/self.bg_size)
                print("total batch: ", h_batch, ", batch group num: ", bg_num , ", batch group size: ", self.bg_size)

                if self.opt.approx:
                    for bg in range (bg_num):
                        print("bg num: ", bg)
                        w_cur_layer = self.sigma_quant_net[l].weight
                        
                        ## considering input pruning to weight
                        w_pruned_cur_layer = w_cur_layer * w_nxt_masks[bg]
                      
                        width = w_cur_layer[w_nxt_masks[bg]].shape[0]       ## ich directional
                        height = w_cur_layer[w_nxt_masks[bg]].shape[1]      ## och directional
                        weight_zero = (w_cur_layer[w_nxt_masks[bg]] == 0).sum()/width/height    ## after pruning, weight sparsity
                        weight_25 = torch.logical_and( (w_cur_layer[w_nxt_masks[bg]] < 2**-1), (w_cur_layer[w_nxt_masks[bg]] > 2**-4))
                        weight_25_count = (weight_25 == 1).sum()/width/height
                        
                        
                        self.sigma_pruned_net[l].weight.data = w_pruned_cur_layer               ## pruned weight @ batch group 
                        #pdb.set_trace()
    
                        ##### Weight Reordering #####
                        w_cur_layer = w_cur_layer.T                                 ## weight w/o pruning (ich x och)
                        weight = w_cur_layer[w_nxt_masks[bg]]                       ## weight after pruning (pruned_ich x och)
    
                        weight_ich_prun = torch.sum(weight==0, dim=1)
                        weight_remain_ich = weight_ich_prun != weight.shape[1]      ## find coarse weight sparsity
                        weight = weight[weight_remain_ich]
                        
                        #### Input slice
                        h_bg_slice = h_prun[bg*self.bg_size : bg*self.bg_size + self.bg_size]       ## bg(batch) x och(nxt layer ich) ; layer input
                        h_prun_slice = h_bg_slice.T
                        h_prun_slice = h_prun_slice[ w_nxt_masks[bg] ]
                        h_prun_slice = h_prun_slice.T
                        
                        ## weight bit sparsity
                        weight_not_zero = weight[weight!=0]                             ## weight bitmap corresponding non-zero weight
                        weight_smallest_unit = torch.min(torch.abs(weight_not_zero))    ## get smallest non-zero weight value
                        weight_q = weight/weight_smallest_unit                          ## weight quantization
                        weight_bin = dec2bin(weight_q.int(), self.opt.w_fxp_bw)         ## get binary weight @ weight quantization
                        weight_inv = torch.logical_not(weight_bin)                      ## make weight 0-bit to 1
                        weight_zero_bit_cnt = torch.sum(weight_inv, dim=2)              ## count weight zero bit
                        weight_zero_bit_ich = torch.sum(weight_zero_bit_cnt, dim=1)     ## sum weight zero bit along w/ ich direction
    
                        weight_zero_bit_cnt_sort = weight_zero_bit_cnt.clone().detach()
                        weight_bin_dup = weight_bin.clone().detach()
                        sort_val, sort_idx = torch.sort(weight_zero_bit_ich, descending=True)       ## weight bit sparsity along w/ ich 
                        pdb.set_trace()


                        ## input bit sparsity
                        for  b in range( h_bg_slice.shape[0]) :
                            batch_data = h_prun_slice[ b ]
    
                            if torch.all( batch_data == 0 ):
                                input_q = batch_data
                            else:
                                #input_q = batch_data/torch.min( torch.abs( batch_data[ batch_data!=0 ] ) )
                                input_q = batch_data/torch.min( torch.abs( h_prun_slice[ h_prun_slice!=0 ] ) )
    
                            if b == 0:
                                input_quantized = input_q.int().reshape(1,-1)
                            else:
                                input_quantized = torch.cat( [input_quantized, input_q.int().reshape(1,-1)], dim=0 )
                        
                        pdb.set_trace() 
                        input_bin = dec2bin( input_quantized, self.opt.i_fxp_bw)
                        input_bin_inv = torch.logical_not( input_bin )
                        input_zero_bit_cnt = torch.sum( input_bin_inv, dim=2 )
                        
    
                        if sort_idx.shape[0]-int(sort_idx.shape[0]/4)*4 != 0:
                            #group_num = int(sort_idx.shape[0]/4) + 1
                            group_num = int(sort_idx.shape[0]/4)
                        else:
                            group_num = int(sort_idx.shape[0]/4)
    
                        top_4k = sort_idx[0:group_num*2]
    
                        top_4k_slice = top_4k[0:group_num]
                        zero_max_list = weight_zero_bit_ich[top_4k_slice]
    
                        #weight_bin_dup = delete_row_tensor(weight_bin_dup, top_4k_slice)
    
                        #if bg == 12: pdb.set_trace()
    
                        
                        ## assign 4k th weight to high bit sparsity weight
                        for ich in range (0, group_num):
                            weight_zero_bit_cnt_sort[ich*4,:] = weight_zero_bit_cnt[top_4k_slice[ich],:]
                            #input_zero_bit_cnt[ich*4, :] = input_zero_bit_cnt[ top_4k_slice[ich], : ]
                            input_zero_bit_cnt[:, ich*4] = input_zero_bit_cnt[ :, top_4k_slice[ich] ]
                        
                            similarity = torch.logical_and(weight_bin[top_4k_slice[ich]], weight_bin_dup)
    
                            match_idx = torch.argmin(torch.sum(torch.sum(similarity, dim=2), dim=1))
    
                            weight_zero_bit_cnt_sort[ich*4+1,:] = weight_zero_bit_cnt[match_idx,:]
                            #input_zero_bit_cnt[ich*4+1, :] = input_zero_bit_cnt[ match_idx, : ]
                            input_zero_bit_cnt[:, ich*4+1] = input_zero_bit_cnt[ :, match_idx ]
    
                        #pdb.set_trace()
    
                        for och in range (0, weight_zero_bit_cnt.shape[1], 32):
    
                            weight_zero_bit_cnt_och32 = torch.sum(weight_zero_bit_cnt_sort[:,och:och+32], dim=1)
                            
                            if och == 0:
                                weight_zero_bit_cnt_sort_och32 = weight_zero_bit_cnt_och32.reshape(-1,1)
    
                            else:
                                weight_zero_bit_cnt_sort_och32 = torch.cat([weight_zero_bit_cnt_sort_och32, weight_zero_bit_cnt_och32.reshape(-1,1)], dim=1)
    
    
                        #pdb.set_trace()
    
                        if sort_idx.shape[0]-int(sort_idx.shape[0]/4)*4 != 0:
                            weight_loop = int(sort_idx.shape[0]/4)
                        else:
                            weight_loop = int(sort_idx.shape[0]/4)
    
    
                        #pdb.set_trace()
                        
                        ## baseline (before weight shuffling)
                        for loop in range (weight_loop):
                            if loop == 0:
                                non_shuffled_zero = weight_zero_bit_ich[0].reshape(1)
                            else:
                                non_shuffled_zero = torch.cat([non_shuffled_zero, weight_zero_bit_ich[loop*4].reshape(1)], dim=0)
    
                        #pdb.set_trace()
                        ### weight shuffling (och32)
                        for ich_och32 in range(0, group_num):#weight_zero_bit_cnt_sort_och32.shape[0], 4):
                            ich_zero_cnt = 0
                            input_zero_cnt = 0
    
                            for och_och32 in range (weight_zero_bit_cnt_sort_och32.shape[1]):
                                ## find high weight bit sparsity ich @ och
                                w_max_element, w_max_idx = torch.max(weight_zero_bit_cnt_sort_och32[4*ich_och32:4*ich_och32+4, och_och32], dim=0)                            
                                #i_max_element, i_max_idx = torch.max(input_zero_bit_cnt[4*ich_och32:4*ich_och32+4, och_och32], dim=0)
                                i_element_w_max_idx = input_zero_bit_cnt[:, 4*ich_och32+w_max_idx]
                                #pdb.set_trace()
    
                                ich_zero_cnt = ich_zero_cnt + w_max_element
                                input_zero_cnt = input_zero_cnt + torch.mean(i_element_w_max_idx.float())*32
    
    
                                #if w_max_idx == 0:
                                #    continue
                                #else:
                                #    weight_zero_bit_cnt_sort_och32[4*ich_och32+w_max_idx, och_och32] = weight_zero_bit_cnt_sort_och32[4*ich_och32, och_och32]
                                #    weight_zero_bit_cnt_sort_och32[4*ich_och32, och_och32] = w_max_element
                                    #weight_zero_bit_cnt_sort_och32[4*ich_och32+w_max_idx, och_och32] = weight_4k
    
    
                            #pdb.set_trace()
                            if ich_och32 == 0:
                                och32_shuffled_list = ich_zero_cnt.reshape(1)
                                input_shuffled_list = input_zero_cnt.reshape(1)
                            else:
                                och32_shuffled_list = torch.cat([och32_shuffled_list, ich_zero_cnt.reshape(1)], dim=0)
                                input_shuffled_list = torch.cat([input_shuffled_list, input_zero_cnt.reshape(1)], dim=0)
    
                        #pdb.set_trace()
                        ##########
                        final = torch.max(och32_shuffled_list,input_shuffled_list)
                        print(l,"th layer ======= Weight Shuffling Perf. =======")
                        print(l,"th layer zero_max idx (before shuffle):            ", non_shuffled_zero, " abs: ", torch.sum(non_shuffled_zero))
                        print(l,"th layer zero_max idx (after och 128 shuffle):     ", zero_max_list, " abs: ", torch.sum(zero_max_list))
                        print(l,"th layer zero_max idx (after och 32 shuffle):      ", och32_shuffled_list, " abs: ", torch.sum(och32_shuffled_list))
                        print(l,"th layer zero_max idx och 128 shuffle performance: ", torch.sum(zero_max_list-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                        print(l,"th layer zero_max idx och 32 shuffle performance:  ", torch.sum(och32_shuffled_list-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                        print("\n",l,"th layer ======= Input Shuffling Perf. =======")
                        print(l,"th layer och32 input feeding                    :  ", input_shuffled_list, " abs: ", torch.sum(input_shuffled_list))
                        print("\n",l,"th layer ======= Weight+Input Total Shuffling Perf. =======")
                        print(l,"th layer och32 shuffle+input feeding            :  ", final, " abs: ", torch.sum(final))
                
                        print(l,"th layer och32 shuffle+input feeding performance:  ", torch.sum(final-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                        print("Batch = ", h_prun_reorder.shape[0], ", ICH = ", h_prun_reorder.shape[1], ", OCH = ", 128, ", Total Weight Zero Bit = ", h_prun_reorder.shape[1]*128*4)
    
                        #### BG Inference
                        pdb.set_trace()
                        h_slice = self.sigma_pruned_net[l](h_bg_slice)
    
                        if bg == 0:
                            h_slice_cat = h_slice
                        else:
                            h_slice_cat = torch.cat([h_slice_cat, h_slice], dim=0)                
    
                    h_prun = h_slice_cat

                    if l != self.num_layers-1:
                        if not(self.opt.train):
                            h_prun = F.relu(h_prun, inplace=True)

                    if self.opt.i_fxp:
                        i_int_bw = int(math.log2(h_prun.max()))
                        h_prun = custom_precision( h_prun, i_int_bw, self.opt.i_fxp_bw-1-i_int_bw, 'fxp')
                else:
                    if not(self.opt.train):
                        h_prun = self.sigma_net[l](save_h)
                    
                    if l != self.num_layers-1:
                        if not(self.opt.train):
                            h_prun = F.relu(h_prun, inplace=True)
                    
                    if self.opt.i_fxp:
                        i_int_bw = int(math.log2(h_prun.max()))
                        h_prun = custom_precision( h_prun, i_int_bw, self.opt.i_fxp_bw-1-i_int_bw, 'fxp')


            #### Activation ####
            #if l != self.num_layers - 1:
            #    h = F.relu(h, inplace=True)

            #    if not(self.opt.train):
            #        h_prun = F.relu(h_prun, inplace=True)
            #        h_quant = F.relu(h_quant, inplace=True)
            

            #### SR batch group pruning ratio ####
            if self.opt.off_loading:
                bg_num = int(h_batch / self.bg_size)
    
    
                w_nxt_masks = []
    
                if l == 4:
                    p_ratio = 0.2
                else:
                    p_ratio = 0.2
    
                for bg in range (bg_num):
                    ### next layer input data pruning
                    h_prun_slice = h_prun[bg*self.bg_size:bg*self.bg_size+self.bg_size]
                    h_prun_slice_mean = h_prun_slice.mean(0)
                    #h_prun_slice_mean_bool = h_prun_slice_mean != 0
                    
                    ### Pruning threshold decision
                    h_prun_slice_mean_max = h_prun_slice_mean.max()
                    h_prun_slice_th = h_prun_slice_mean_max * p_ratio                           # threshold = ich value * p_ratio
                    h_prun_slice_important_ich = h_prun_slice_mean >= h_prun_slice_th
                    h_bg_mask = h_prun_slice_important_ich                
    
                    h_bg_prun_ratio = (h_bg_mask == 0).sum()/128
                    #print(l,"th layer ", bg, "th batch group pruned ratio: ",h_bg_prun_ratio)
    
    
                    w_nxt_masks.append(h_bg_mask)
    
                    if bg == 0:
                        bg_masks = h_bg_mask
                        bg_prun_ratio_list = h_bg_prun_ratio.reshape(1)
                    else:
                        bg_masks = torch.cat([bg_masks, h_bg_mask], dim=0)      # cat batch group * ich
                        bg_prun_ratio_list = torch.cat([bg_prun_ratio_list, h_bg_prun_ratio.reshape(1)], dim=0)
                
    
                h_prun_batch, h_prun_ich = h_prun.shape
                bg_masks = bg_masks.reshape(-1, h_prun_ich)
                bg_masks_clone = bg_masks.clone().detach()
                bg_masks_origin = bg_masks.clone().detach()     # batch group wise pruning masks
    
                print(l,"th layer batch group pruned ratio: ",torch.mean(bg_prun_ratio_list))
                print(l,"th layer coarse sparsity: ", (torch.sum(h_prun, dim=1) == 0).sum(), " / ", h_prun_batch)
                print(l,"th layer sparsity       : ", (h_prun == 0).sum(), " / ", h_prun_batch*h_prun_ich , " = ", (h_prun == 0).sum()/h_prun_batch/h_prun_ich)
    
                ####  Batch Group Sim  ####
                bg_num = bg_masks_origin.shape[0]
                ich    = bg_masks_origin.shape[1]
                sim_bg_unit = 125
                
                bg_rand_idx = torch.randperm(bg_num)
                bg_masks_shuffle = bg_masks_origin.clone().detach()
                bg_masks_shuffle = bg_masks_shuffle[bg_rand_idx]
                
                for bg_sim_ptr in range (0, bg_num, 2):
                    prior_bg = bg_masks_shuffle[bg_sim_ptr]
                    nxt_bg   = bg_masks_shuffle[bg_sim_ptr+1]
                    w_change_mask = torch.logical_xor(prior_bg, nxt_bg)
                    bg_dif = (w_change_mask == 1).sum()
                    sram_fetch = (nxt_bg == 1).sum()
                    if bg_sim_ptr == 0:
                        bg_dif_list = bg_dif.reshape(1)
                        sram_fetch_list = sram_fetch.reshape(1)
                    else:
                        bg_dif_list = torch.cat([bg_dif_list, bg_dif.reshape(1)], dim=0)
                        sram_fetch_list = torch.cat([sram_fetch_list, sram_fetch.reshape(1)], dim=0)
    
                sram_fetch_ratio_list = sram_fetch_list/(ich*(1-torch.mean(bg_prun_ratio_list)))
    
                for i in range (int(bg_num/2)):
                    if sram_fetch_ratio_list[i] > 1:
                        sram_fetch_ratio_list[i] = 1
                
                sim = 1-sram_fetch_ratio_list
    
                for j in range (0, int(bg_num/2), sim_bg_unit):
                    sim_bg125 = torch.mean(sim[j:j+sim_bg_unit])
    
                    if j==0:
                        sim_bg125_list = sim_bg125.reshape(1)
                    else:
                        sim_bg125_list = torch.cat([sim_bg125_list, sim_bg125.reshape(1)], dim=0)
    
                print(l,"th layer batch group", sim_bg_unit," similarity  : ",torch.mean(sim_bg125_list.float()))
    
    
              
                ####    OFF-LOADING    ####
                #### 2 entity matching ####
                print(l, "th layer before off-loading util: ", 1-torch.mean(bg_prun_ratio_list))
                entity2_match_idx = torch.linspace(start=0, end=bg_masks.shape[0]-1, steps=bg_masks.shape[0], dtype=torch.int64)
                entity2_match_idx = torch.cat([entity2_match_idx, entity2_match_idx], dim=0).reshape(2,-1)
                for ptr in range (bg_num):
                    new_bg_num = bg_masks.shape[0]
                    ich = bg_masks.shape[1]
    
                    bg_masks_target = bg_masks[ptr]
                    bg_masks_slice = torch.concat([bg_masks[0:ptr], bg_masks[ptr+1:new_bg_num]], dim=0)
    
                    bg_masks_util_vec = torch.logical_or(bg_masks_target, bg_masks_slice)   # comp HW uti
                    bg_masks_util = torch.sum(bg_masks_util_vec, axis=1)    # batch directional summation
                    bg_masks_max_idx = torch.argmax(bg_masks_util)
                    bg_masks_util_max = torch.max(bg_masks_util)
    
                    if bg_masks_max_idx >= ptr:
                        bg_masks_match_idx = bg_masks_max_idx+1
                    else:
                        bg_masks_match_idx = bg_masks_max_idx
    
                    
                    #print("batch group ", ptr, " util : ", bg_masks_util_max.cpu().detach()/ich, ", with bg_masks[", bg_masks_match_idx,"]")
                    
                    # reset bg_masks 
                    bg_masks[ptr] = False
                    bg_masks[bg_masks_match_idx] = False
    
                    # update bg_masks_clone
                    bg_masks_clone[ptr] = bg_masks_util_vec[bg_masks_max_idx]
                    bg_masks_clone[bg_masks_match_idx] = False
    
                    # store idx
                    entity2_match_idx[1][ptr] = bg_masks_match_idx
    
                    if ptr == 0:
                        bg_2entity_util = (bg_masks_util_max/ich).reshape(1)
                    else:
                        bg_2entity_util = torch.cat([bg_2entity_util,(bg_masks_util_max/ich).reshape(1)], dim=0)
    
    
                bg_2entity_min_util = torch.min(bg_2entity_util)
                bg_2entity_max_util = torch.max(bg_2entity_util)
                print(l,"th layer bg2 off-loading min util: ", bg_2entity_min_util,", max util: ", bg_2entity_max_util)
    
                #### 4 entitiy matching ####
                entity4_match_idx = torch.cat([entity2_match_idx, entity2_match_idx], dim=0)
                for ptr in range (bg_num):
                    new_bg_num = bg_masks_clone.shape[0]
                    ich = bg_masks_clone.shape[1]
    
                    bg_masks_target = bg_masks_clone[ptr]
                    bg_masks_slice = torch.concat([bg_masks_clone[0:ptr], bg_masks_clone[ptr+1:new_bg_num]], dim=0)
    
                    bg_masks_util_vec = torch.logical_or(bg_masks_target, bg_masks_slice)   # comp HW util
                    bg_masks_util = torch.sum(bg_masks_util_vec, axis=1)    # batch directional summation
                    bg_masks_max_idx = torch.argmax(bg_masks_util)
                    bg_masks_util_max = torch.max(bg_masks_util)
                    #print("batch group ", ptr, " util : ", bg_masks_util_max.cpu().detach()/ich, ", with bg ", bg_masks_max_idx)
    
                    if bg_masks_max_idx >= ptr:
                        bg_masks_match_idx = bg_masks_max_idx+1
                    else:
                        bg_masks_match_idx = bg_masks_max_idx
    
                    entity4_match_idx[2][ptr] = entity2_match_idx[0][bg_masks_match_idx]
                    entity4_match_idx[3][ptr] = entity2_match_idx[1][bg_masks_match_idx]
    
                    bg_masks_clone[ptr] = False
                    bg_masks_clone[bg_masks_max_idx] = False
    
                    if ptr == 0:
                        bg_4entity_util = (bg_masks_util_max/ich).reshape(1)
                    else:
                        bg_4entity_util = torch.cat([bg_4entity_util,(bg_masks_util_max/ich).reshape(1)], dim=0)
    
                bg_4entity_min_util = torch.min(bg_4entity_util)
                bg_4entity_max_util = torch.max(bg_4entity_util)
                print(l, "th layer bg4 off-loading min util: ", bg_4entity_min_util,", max util: ", bg_4entity_max_util)               
    
                #### batch group 4 allocate core utilization ####
                entity4_ich = bg_masks_origin.shape[1]
                entity4_batch = bg_masks_origin.shape[0]
    
                entity4_h = entity4_match_idx.shape[0]
                entity4_w = entity4_match_idx.shape[1]
    
                
                for bg in range (entity4_w):
                    for h_ptr in range (entity4_h):
                        ptr = entity4_match_idx[h_ptr][bg]
                        if h_ptr == 0 :
                            no_access = torch.logical_not(bg_masks_origin[ptr])
                            hot = bg_masks_origin[ptr]
                            intersection = bg_masks_origin[ptr].int()
                        else:
                            no_access = torch.logical_and(no_access, torch.logical_not(bg_masks_origin[ptr]))
                            hot = torch.logical_and(hot, bg_masks_origin[ptr])
                            intersection = intersection + bg_masks_origin[ptr].int()
    
                    intersection = intersection - 1
                    intersec_group2 = ((intersection > 0) == 1).sum()
                    inter = torch.tensor(0, dtype=torch.int32)
                    for ich in range (intersection.shape[0]):
                        if intersection[ich] > 0:
                            inter = inter + intersection[ich]
                    no_access = (no_access == 1).sum()
                    hot = (hot == 1).sum()
                    #print("no_access: ", no_access, ", hot: ", hot, ", 2 group: ", (intersection.shape[0]-no_access-hot), ", intersection: ", inter)
    
                    if bg == 0:
                        no_access_list = no_access.reshape(1)
                        hot_list = hot.reshape(1)
                        group2_list = (intersection.shape[0]-no_access-hot).reshape(1)
                        inter_list = inter.reshape(1)
                        inter_nodup_list = intersec_group2.reshape(1)
                    else:
                        #if l == 4 and (bg == 15873 or bg == 15872):
                        #    print("layer4 bg: ", bg)
                        #    pdb.set_trace()
                        no_access_list = torch.cat([no_access_list, no_access.reshape(1)], dim=0)
                        hot_list = torch.cat([hot_list, hot.reshape(1)], dim=0)
                        group2_list = torch.cat([group2_list, (intersection.shape[0]-no_access-hot).reshape(1)], dim=0)
                        inter_list = torch.cat([inter_list, inter.reshape(1)], dim=0)
                        inter_nodup_list = torch.cat([inter_nodup_list, intersec_group2.reshape(1)], dim=0)
                
                
                no_access_list_mean = torch.mean(no_access_list.float())
                hot_list_mean = torch.mean(hot_list.float())
                group2_list_mean = torch.mean(group2_list.float())
                inter_list_mean = torch.mean(inter_list.float())
                inter_nodup_list_mean = torch.mean(inter_nodup_list.float())
    
                print(l,"th layer gc core] no access: ", no_access_list_mean, ", hot: ", hot_list_mean, ", group2: ", group2_list_mean, ", intersection (dup O): ", inter_list_mean, ", intersection (dup X): ", inter_nodup_list_mean)       # gc core utilization
    
    
    
                ####      RECONFIG SRAM READ/WRITE     ####
                base_core_num = 5
                bg_masks_sram_access = bg_masks_shuffle.clone().detach()
    
                iteration = int(bg_num/base_core_num)
                for i in range (0, bg_num, base_core_num*2):
                    core0     =   (bg_masks_sram_access[i+5] == 1).sum()
                    core1     =   (bg_masks_sram_access[i+6] == 1).sum()
                    core2     =   (bg_masks_sram_access[i+7] == 1).sum()
                    core3     =   (bg_masks_sram_access[i+8] == 1).sum()
                    core4     =   (bg_masks_sram_access[i+9] == 1).sum()
    
                    if i==0:
                        core0_sram_access = core0.reshape(1)
                        core1_sram_access = core1.reshape(1)
                        core2_sram_access = core2.reshape(1)
                        core3_sram_access = core3.reshape(1)
                        core4_sram_access = core4.reshape(1)
                    else:
                        core0_sram_access = torch.cat([core0_sram_access, core0.reshape(1)], dim=0)
                        core1_sram_access = torch.cat([core1_sram_access, core1.reshape(1)], dim=0)
                        core2_sram_access = torch.cat([core2_sram_access, core2.reshape(1)], dim=0)
                        core3_sram_access = torch.cat([core3_sram_access, core3.reshape(1)], dim=0)
                        core4_sram_access = torch.cat([core4_sram_access, core4.reshape(1)], dim=0)
    
                sram_access = torch.sum(core0_sram_access.float()) + torch.sum(core1_sram_access.float()) + torch.sum(core2_sram_access.float()) + torch.sum(core3_sram_access.float()) + torch.sum(core4_sram_access.float())    # baseline sram access = RF write
                print(l,"th layer baseline sram access: ", sram_access, "rf write: ", sram_access)
                print(l,"th layer proposed rf write: ", (hot_list_mean*3+inter_list_mean)*core0_sram_access.shape[0])
                print(l,"th layer basedline = ", sram_access/((hot_list_mean*3+inter_list_mean)*core0_sram_access.shape[0]), " x proposed")
                print("====================================")

            #pdb.set_trace()

            #### SR batch directional weight pruning ####
            #if l != self.num_layers - 1:
            #   
            #    h_mean = h.mean(0)
            #    #h_mean = np.mean(h.detach().cpu().numpy(), axis = 0) 
            #    print(str(l)+"th layer mean value: ",h_mean)
            #    

            #    #h_max_of_mean = np.max(h_mean)
            #    h_max_of_mean = h_mean.max()
            #    print(str(l)+"th layer max of mean value: ",h_max_of_mean)
            #    h_pruning_th = h_max_of_mean*self.sigma_net_prun_th_ratio[l]
            #    print(str(l)+"th layer pruning threshold: ",h_pruning_th)


            #    h_important_ich = h_mean >= h_pruning_th
            #    print(str(l)+"th layer important ich value: ",h_important_ich)

            #    w_mask = h_important_ich
            #   
            #    
            #    w_nxt_layer = self.sigma_pruned_net[l+1].weight
            #    origin_w_nxt_layer = self.sigma_net[l+1].weight 
            #    print(str(l+1)+" nxt w:", w_nxt_layer)
            #    w_pruned_nxt_layer = w_nxt_layer * w_mask
            #    print(str(l+1)+" nxt pruned w:", w_pruned_nxt_layer)
            #    print(str(l+1)+" original nxt pruned w:", origin_w_nxt_layer)
            #    #self.sigma_pruned_net[l+1].weight.data = torch.Tensor(w_pruned_nxt_layer)
            #    self.sigma_pruned_net[l+1].weight.data = w_pruned_nxt_layer


            #### SR output data log #### 
            #with open (outputfile_name, 'w', newline='') as f:
            #    writer = csv.writer(f)
            #    writer.writerows(h.cpu().numpy())
            #    torch.save(h,"./intermediate/sigmaNet_layer" + str(l) + "_origin_output.pth")

            #with open (prun_outputfile_name, 'w', newline='') as f:
            #    writer = csv.writer(f)
            #    writer.writerows(h_prun.cpu().numpy())
            #    torch.save(h_prun,"./intermediate/sigmaNet_layer" + str(l) + "_prun_output.pth")               

        #sigma = F.relu(h[..., 0])

        sigmafile_name = "./intermediate/sigmaNet_layer" + str(l) + "_origin_sigma.csv"
        prun_sigmafile_name = "./intermediate/sigmaNet_layer" + str(l) + "_prun_sigma.csv"
        print(h.shape)
        sigma = trunc_exp(h[..., 0])

        if not(self.opt.train):
            sigma_prun = trunc_exp(h_prun[..., 0])
            sigma_quant = trunc_exp(h_quant[..., 0])
        else:
            sigma_prun = None
            sigma_quant = None

        #### SR sigma data log #### 
        #with open (sigmafile_name, 'w', newline='') as f:
        #    writer = csv.writer(f)
        #    for item in sigma:
        #        writer.writerow(sigma.cpu().numpy())
        #        torch.save(h,"./intermediate/sigmaNet_layer_origin_sigma.pth")

        #with open (prun_sigmafile_name, 'w', newline='') as f:
        #    writer = csv.writer(f)
        #    for item in sigma:
        #        writer.writerow(sigma_prun.cpu().numpy())
        #        torch.save(h,"./intermediate/sigmaNet_layer_prun_sigma.pth")


        #### SR pruned weight & original weight log ####
        #for l in range(self.num_layers):
        #    origin_wfile_name = "./weight/sigmaNet_layer" + str(l) + "_origin_weight.csv"
        #    pruned_wfile_name = "./weight/sigmaNet_layer" + str(l) + "_pruned_weight.csv"

        #    with open (origin_wfile_name, 'w', newline='') as f:
        #        writer = csv.writer(f)
        #        writer.writerows(np.abs(self.sigma_net[l].weight.data.cpu().numpy()))
        #        torch.save(self.sigma_net[l].weight.data,"./weight/sigmaNet_layer" + str(l) + "_origin_weight.pth")

        #    with open (pruned_wfile_name, 'w', newline='') as f:
        #        writer = csv.writer(f)
        #        writer.writerows(np.abs(self.sigma_pruned_net[l].weight.data.cpu().numpy()))
        #        torch.save(self.sigma_pruned_net[l].weight.data,"./weight/sigmaNet_layer" + str(l) + "_pruned_weight.pth")

        geo_feat = h[..., 1:]
        #pdb.set_trace()

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'sigma_prun': sigma_prun,
            'sigma_quant' : sigma_quant
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def delete_row_tensor(a, del_row):
    device = a.device
    n = a.cpu().detach().numpy()
    del_row = del_row.cpu().detach().numpy()
    n = np.delete(n, del_row, 0)

    n = torch.from_numpy(n).to(device)
    return n

class Quant(Function):
    @staticmethod
    def forward(ctx, x, w, i_fxp, i_fxp_bw, err_fxp, err_fxp_bw, l):
        print("Quant FF "+str(l)+" Start")
        if i_fxp and l>0 and l<5:
            i_int_bw = int(math.log2(x.max()))
            x = custom_precision(x, i_int_bw, i_fxp_bw-1-i_int_bw, 'fxp')
        
        out = torch.mm(x, w)
        ctx.constant = (x, w, err_fxp, err_fxp_bw, l)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w, err_fxp, err_fxp_bw, l = ctx.constant
        print("Quant BP" + str(l) +" Start")
        w = w.type(torch.cuda.HalfTensor)
        x = x.type(torch.cuda.HalfTensor)
        grad_output = grad_output.type(torch.cuda.HalfTensor)
        
        if l<5 and err_fxp and l>0:
            
            if torch.any(torch.isnan(grad_output)):
                grad_output = grad_output.nan_to_num(nan=0.0)
            if torch.any(torch.isinf(grad_output)):
                grad_output = grad_output.nan_to_num(posinf=0.0, neginf=0.0)
            
            err_int_bw = int(math.log2(grad_output.max()))
            grad_output = custom_precision(grad_output, err_int_bw, err_fxp_bw-1-err_int_bw, 'fxp')
            grad_output = grad_output.type(torch.cuda.HalfTensor)

        err = torch.mm(grad_output, w.T)
        w_grad = torch.mm(x.T, grad_output)
        
        del ctx
        torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()
        del w
        torch.cuda.empty_cache()
        del grad_output
        torch.cuda.empty_cache()

        #return err, w_grad, None, None, None, None, None
        return err, None, None, None, None, None, None


class QFC(nn.Linear):
    def __init__(self, in_features, out_features, bias, i_fxp, i_fxp_bw, err_fxp, err_fxp_bw, l):
        super().__init__(in_features, out_features, bias)
        self.i_fxp = i_fxp
        self.i_fxp_bw = i_fxp_bw
        self.err_fxp = err_fxp
        self.err_fxp_bw = err_fxp_bw
        self.l = l

    def forward(self, x):
        out = Quant.apply(x, self.weight.T, self.i_fxp, self.i_fxp_bw, self.err_fxp, self.err_fxp_bw, self.l) 
        return out

class AQFC(nn.Linear):
    def __init__(self, in_features, out_features, bias, w_fxp_bw, i_fxp, i_fxp_bw, err_fxp, err_fxp_bw, l, at_fan_in, compressor_accuracy):
        super().__init__(in_features, out_features, bias)
        self.w_fxp_bw = w_fxp_bw
        self.i_fxp = i_fxp
        self.i_fxp_bw = i_fxp_bw
        self.err_fxp = err_fxp
        self.err_fxp_bw = err_fxp_bw
        self.l = l
        self.at_fan_in = at_fan_in
        self.compressor_accuracy = compressor_accuracy

    def forward(self, x):
        out = AQuant.apply(x, self.weight.T, self.w_fxp_bw, self.i_fxp, self.i_fxp_bw, self.err_fxp, self.err_fxp_bw, self.l, self.at_fan_in, self.compressor_accuracy) 
        return out

class AQuant(Function):
    def shuffling_mm(ctx, x, w, i_fxp_bw, w_fxp_bw):
        ### Check weight bit sparsity
        w_nonzero_idx = w != 0
        w_smallest_unit = torch.min( torch.abs(w[w_nonzero_idx]) )
        w_q = w.T/w_smallest_unit                                                                   # och, ich
        w_bin = dec2bin( w_q.int(), w_fxp_bw )                                                      # och, ich, 4
        w_accurate_flag = w_bin[:,:,0:4].sum(dim=2)                                                 # och, ich
        w_tiled = w_accurate_flag.reshape(-1, 32, w_accurate_flag.shape[1])                         # och tiled num, 32, ich
        w_tiled = w_tiled.transpose(1,2)                                                            # och tiled num, ich, 32 (och unit)
        w_shuffle_flag = w_tiled.sum(dim=2)                                                         # accum och direction; [och tiled num, ich]
        w_shuffle_flag = w_shuffle_flag.reshape(-1, int(w_shuffle_flag.shape[1]/4), 4)              # och tiled num, compressor num, 4
        w_shuffled = w_shuffle_flag.reshape(-1, int(w_tiled.shape[1]/32), 8, 4)                     # och tiled num, at num, compressor num, compressor fan-in
        sort_val, sort_idx = torch.sort( w_shuffled, dim=3, descending=False )                      # och tiled num, compressor num, 4 (compressor fan-in)

        del sort_val
        torch.cuda.empty_cache()

        W_tiled = w.T.reshape(-1, 32, int(w.shape[0]/32), 8, 4)                                     # och tiled num, 32(och unit), at_num, 32(at fan-in)
        #w_transform = w.T.clone().reshape(-1, int(w_accurate_flag.shape[1]/4), 4)

        for och_num in range (W_tiled.shape[0]):
            for at_num in range (W_tiled.shape[2]):
                for compressor_num in range (W_tiled.shape[3]):
                    #pdb.set_trace()
                    W_tiled[och_num, :, at_num, compressor_num] = W_tiled[och_num, :, at_num, compressor_num, sort_idx[och_num, at_num, compressor_num]]

        for och_unit in range (sort_idx.shape[0]):
            for at_num in range (sort_idx.shape[1]):
                for compressor_unit in range (sort_idx.shape[2]):
                    a = 32*at_num + 4*compressor_unit + sort_idx[och_unit, at_num, compressor_unit]

                    if compressor_unit == 0:
                        A = a
                    else:
                        A = torch.cat( [A, a] )
                if at_num == 0:
                    B = A
                else:
                    B = torch.cat( [B, A] )
            if och_unit == 0:
                och_unit_sort_idx = B.reshape(1, -1)
            else:
                och_unit_sort_idx = torch.cat( [och_unit_sort_idx, B.reshape(1, -1)], dim=0 )
        
#        pdb.set_trace()

        x_nz_idx = x != 0
        x_nz_min = torch.min( torch.abs(x[ x_nz_idx ]) )
        x_q = (x / x_nz_min).int()

        w_nz_idx = W_tiled != 0
        w_nz_min = torch.min( torch.abs( W_tiled[w_nz_idx] ) )
        w_tiled_q = (W_tiled / w_nz_min).int()

        #pdb.set_trace()

        for och_unit in tqdm(range (W_tiled.shape[0])):
            ##################### golden ######################
            # float for golden check
            x_och_unit = x[ :, och_unit_sort_idx[och_unit] ]
            W_och32_tiled = W_tiled[och_unit].reshape(32, -1).T
            out_och32_tiled = torch.mm( x_och_unit, W_och32_tiled )     ## golden

            if och_unit == 0:
                out = out_och32_tiled
            else:
                out = torch.cat( [out, out_och32_tiled], dim=1 )
            ####################################################

            # W (ich * 32)
            x_q_unit  = x_q[ :, och_unit_sort_idx[och_unit] ]
            w_q_tiled = w_tiled_q[och_unit]
            
            for batch in tqdm(range( x_q_unit.shape[0] )):
                ## bs_control * bs_operand
                #pdb.set_trace()
                bs_control = dec2bin(x_q_unit[batch].reshape(-1, 8, 4), i_fxp_bw)
                bs_operand = dec2bin(w_q_tiled  , w_fxp_bw)
    
                ## bit plane for compressor accuracy
                #print("input bit: ", bs_control)
                #print("==========================")
                for i_bs in reversed(range(i_fxp_bw)):
                    ## input bit serial
                    bs_step = bs_control[ :, :, :, i_bs ]                                                                       # input bit @ bs
                    #print("input bs: ", bs_step)
                    bs_operand_step = bs_operand                                                                                # weight
    
                    at, compressor_num, fan_in = torch.nonzero( torch.logical_not( bs_step ).int(), as_tuple=True )             # find input 0-bit 
                    
                    bs_operand_step[:,at,compressor_num,fan_in,:] = 0                                                           # weight 0 @ input 0-bit 
    
                    A = bs_operand_step.transpose(0,4)
                    B = A.transpose(1,4)
                    C = B.transpose(2,4)
                    D = C.transpose(3,4)                                                                                        # bit plane, och unit(32), at num, compressor num, compressor fan in                
    
                    weight_shaped = bs_operand_step.reshape(32,128,w_fxp_bw).int()                                              # weight 32x128x4b
                    sign = weight_shaped[:,:,0].reshape( weight_shaped.shape[0], weight_shaped.shape[1], 1 )                    # 2's complement weight --> dec
                    sign_och, sign_ich, sign_bit = torch.nonzero( sign, as_tuple=True )                                         # Find - weight
                    weight_shaped = weight_shaped.bool()
                    weight_shaped[ sign_och, sign_ich, 1:4 ] = torch.logical_not( weight_shaped[sign_och, sign_ich, 1:4] )      # invert bit @ - weight
                    #weight_shaped = weight_shaped.sum( dim=2 )
    
                    weight_bit_position_mask = 2**torch.arange(w_fxp_bw-1, -1, -1)
                    weight_bit_position_mask[0] = 1
                    weight_shaped = (weight_bit_position_mask * weight_shaped)
                    weight_shaped = weight_shaped.sum( dim=2 )
    
                    weight_shaped[sign_och, sign_ich] = -weight_shaped[sign_och, sign_ich]                                      # - sign @ minus weight
                    accurate_result_och32 = weight_shaped.sum( dim=1 )
    
                    for at in reversed(range(D.shape[0])):
                        # at
                        at_32way_fan_in = D[at]     # och unit (32), at_num, compressor num, comresspr fan in
    
                        lsb_err = torch.logical_and( at_32way_fan_in[:,:,:,0], at_32way_fan_in[:,:,:,1] ).int()       # at 1st stage lsb err
    
                        #pdb.set_trace()
                        stage2_err_map  = at_32way_fan_in.sum(dim=3).int()
                        #pdb.set_trace()
                        stage2_err_map  = stage2_err_map - lsb_err
                        stage2_err_bmap = dec2bin( stage2_err_map.int(), 2 )
                        #pdb.set_trace()
                        stage2_err_bmap = stage2_err_bmap.transpose(0,3)
                        stage2_err_bmap = stage2_err_bmap.transpose(1,3)        # carry_err_bmap, sum_err_bmap
                        stage2_err_bmap = stage2_err_bmap.transpose(2,3)        # 2 (carry/sum) , och unit (32), at num, compressor num (8)
    
                        stage2_compressor_in = stage2_err_bmap.reshape( 2, 32, 4, 2, 4 )
                        carry_b = stage2_compressor_in[0]
                        sum_b   = stage2_compressor_in[1]
    
                        carry_compressor_err_bmap = torch.logical_and( carry_b[:,:,:,0], carry_b[:,:,:,1] )
                        sum_compressor_err_bmap   = torch.logical_and( sum_b[:,:,:,0], sum_b[:,:,:,1] )
    
                        carry_compressor_err_map  = carry_compressor_err_bmap.sum( dim=2 ).sum( dim=1 ).int()
                        sum_compressor_err_map    = sum_compressor_err_bmap.sum( dim=2 ).sum( dim=1 ).int()
    
                        och32_fan32_accurate_at_sum = at_32way_fan_in.reshape(32,-1).sum( dim=1 )
                        
                        ## 32 fan in 32 och at sum : och32_fan32_accurate_at_sum
                        ## lsb compressor err      : lsb_err
                        ## sum compressor err      : sum_compressor_err_map
                        ## carry compressor err    : carry_compressor_err_map
                        #pdb.set_trace()
    
                        if at == 3:
                            err = -lsb_err.reshape(32,-1).sum( dim=1 )-sum_compressor_err_map-2*carry_compressor_err_map
                        else:
                            err = err + (-lsb_err.reshape(32,-1).sum( dim=1 )-sum_compressor_err_map-2*carry_compressor_err_map)*(2**(3-at))
                        #print("err: ", err)
    
    
                    #print("final err: ", err)
                    #print("accurate result: ", accurate_result_och32)
                    approx_result_och32 = accurate_result_och32 + err
                    
                    if i_bs == 3:
                        bs_accurate_result_och32 = accurate_result_och32
                        bs_approx_result_och32 = approx_result_och32
                    else:
                        bs_accurate_result_och32 = bs_accurate_result_och32 + accurate_result_och32*2**(3-i_bs)
                        bs_approx_result_och32 = bs_approx_result_och32 + approx_result_och32*2**(3-i_bs)
    
                    #print("bs result: ", bs_accurate_result_och32)

                    if batch == 0:
                        batch_accurate_result_och32 = bs_accurate_result_och32.reshape(1,-1)
                        batch_approx_result_och32 = bs_approx_result_och32.reshape(1,-1)
                    else:
                        batch_accurate_result_och32 = torch.cat([ batch_accurate_result_och32, bs_accurate_result_och32.reshape(1,-1)], dim=0)
                        batch_approx_result_och32 = torch.cat([batch_approx_result_och32, bs_approx_result_och32.reshape(1,-1)], dim=0)

            if och_unit == 0:
                batch_accurate_result = batch_accurate_result_och32
                batch_approx_result = batch_approx_result_och32
            else:
                batch_accurate_result = torch.cat([ batch_accurate_result , batch_accurate_result_och32 ], dim=1)
                batch_approx_result = torch.cat([ batch_approx_result , batch_approx_result_och32 ], dim=1)


        pdb.set_trace()




#        for b in range (x.shape[0]):
#            x_slice = x[0].reshape(-1, 8, 4)
#
#            for och_unit in range(W_tiled.shape[0]):
#                for och in range(32):
#                    A = x_slice * W_tiled[och_unit, och]
#            pdb.set_trace()
#            x_slice_broadcast = x_slice.repeat(W_tiled.shape[0],0,0)
#            pdb.set_trace()


#        pdb.set_trace()
        #w_shuffle_flag = w_accurate_flag.reshape(-1, int(w_accurate_flag.shape[1]/4), 4)            # och, compressor_num for 1 ich, 4 
        #sort_val, sort_idx = torch.sort( w_shuffle_flag, dim=2, descending=False )


        del w_q
        torch.cuda.empty_cache()

        ## weight/input sorted
#        activation = x.clone().reshape(-1, int(x.shape[1]/4), 4)
        
#        pdb.set_trace()
#        for och in range(w_transform.shape[0]):
#            for c_num in range(w_transform.shape[1]):
#                #pdb.set_trace()
#                w_transform[och, c_num] = w_transform[och, c_num, sort_idx[och][c_num]]
#                activation[och, c_num]  = activation[och, c_num, sort_idx[och][c_num]]
#        
#        pdb.set_trace()
        del sort_idx
        torch.cuda.empty_cache()
        del w_shuffle_flag
        torch.cuda.empty_cache()
        del w_bin
        torch.cuda.empty_cache()
        del w_accurate_flag
        torch.cuda.empty_cache()

        return out

    @staticmethod
    def forward(ctx, x, w, w_fxp_bw, i_fxp, i_fxp_bw, err_fxp, err_fxp_bw, l, at_fan_in, compressor_accuracy):
        print("Approx Quant FF "+str(l)+" Start")
        if i_fxp and l>0 and l<5:
            i_int_bw = int(math.log2(x.max()))
            x = custom_precision(x, i_int_bw, i_fxp_bw-1-i_int_bw, 'fxp')
        
        ## ground truth
        out = torch.mm(x, w)
        ctx.constant = (x, w, err_fxp, err_fxp_bw, l)

        ## approx comp
        if compressor_accuracy and l>0 and l<5:
            ## shuffling weight
            out_shuffling = AQuant.shuffling_mm(ctx, x, w, i_fxp_bw, w_fxp_bw)
            pdb.set_trace()

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w, err_fxp, err_fxp_bw, l = ctx.constant
        print("Approx Quant BP" + str(l) +" Start")

        w = w.type(torch.cuda.HalfTensor)
        x = x.type(torch.cuda.HalfTensor)
        grad_output = grad_output.type(torch.cuda.HalfTensor)
        
        if l<5 and err_fxp and l>0:
            
            if torch.any(torch.isnan(grad_output)):
                grad_output = grad_output.nan_to_num(nan=0.0)
            if torch.any(torch.isinf(grad_output)):
                grad_output = grad_output.nan_to_num(posinf=0.0, neginf=0.0)
            
            err_int_bw = int(math.log2(grad_output.max()))
            grad_output = custom_precision(grad_output, err_int_bw, err_fxp_bw-1-err_int_bw, 'fxp')
            grad_output = grad_output.type(torch.cuda.HalfTensor)

        err = torch.mm(grad_output, w.T)
        w_grad = torch.mm(x.T, grad_output)
        
        del ctx
        torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()
        del w
        torch.cuda.empty_cache()
        del grad_output
        torch.cuda.empty_cache()

        #return err, w_grad, None, None, None, None, None
        return err, None, None, None, None, None, None


        
