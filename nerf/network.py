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

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
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

        self.fxp = fxp
        self.fxp_bw = fxp_bw
        self.encoding = encoding

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        
        
        self.bg_size = 32
        sigma_net = []
        sigma_pruned_net = []
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
        

        self.sigma_net = nn.ModuleList(sigma_net)
        self.sigma_pruned_net = nn.ModuleList(sigma_pruned_net)

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
        pdb.set_trace()
        for l in range(self.num_layers):
            #### SR Quantization ####
            if self.fxp and l>0 and l<self.num_layers-1:
                if h.max()<=1 :
                    i_int_bw = 0
                else:
                    i_int_bw = math.log2(h.max())
                h = custom_precision(h, i_int_bw, self.fxp_bw-1-i_int_bw, 'fxp')
                
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        pdb.set_trace()
        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

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

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        print(self.sigma_net)
        print("nerwork.py (151) density func")
        if self.encoding == 'frequency':
            x = self.encoder(x)
        elif self.encoding == 'hashgrid':
            x = self.encoder(x, bound=self.bound)


        h = x
        h_prun = x

        h_batch, h_ich = h.shape

        for l in range(self.num_layers):
            print(l,"th layer inference")

            inputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_origin_input.csv"
            outputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_origin_output.csv"
            prun_inputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_prun_input.csv"
            prun_outputfile_name = "./intermediate/sigmaNet_layer" + str(l) + "_prun_output.csv"

            #### SR Quantization ####
            if self.fxp and l>0 and l<self.num_layers-1:
                i_int_bw = int(math.log2(h.max()))
                h = custom_precision(h, i_int_bw, self.fxp_bw-1-i_int_bw, 'fxp')
            if self.fxp and l>0 and l<self.num_layers-1:
                i_int_bw = int(math.log2(h_prun.max()))
                h_prun = custom_precision(h_prun, i_int_bw, self.fxp_bw-1-i_int_bw, 'fxp')

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
            if l==0 :

                w_cur_layer = self.sigma_net[l].weight

                weight = w_cur_layer.T


                ### Check ich prun ###
                weight_ich_prun = torch.sum(weight==0, dim=1)
                weight_remain_ich = weight_ich_prun != weight.shape[1]
                weight = weight[weight_remain_ich]

                weight_not_zero = weight[weight!=0]
                weight_smallest_unit = torch.min(torch.abs(weight_not_zero))
                weight_q = weight/weight_smallest_unit
                weight_bin = dec2bin(weight_q.int(), 4)
                weight_inv = torch.logical_not(weight_bin)
                weight_zero_bit_cnt = torch.sum(weight_inv, dim=2)
                weight_zero_bit_ich = torch.sum(weight_zero_bit_cnt, dim=1)

                weight_zero_bit_cnt_sort = weight_zero_bit_cnt.clone().detach()
                weight_bin_dup = weight_bin.clone().detach()
                
                sort_val, sort_idx = torch.sort(weight_zero_bit_ich, descending=True)
                top_4k = sort_idx[0:round(sort_idx.shape[0]/4)*2]

                top_4k_slice = top_4k[0:round(sort_idx.shape[0]/4)]
                zero_max_list = weight_zero_bit_ich[top_4k_slice]

                #weight_bin_dup = delete_row_tensor(weight_bin_dup, top_4k_slice)



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
                h_prun_reorder_bin = dec2bin(input_q.int(), 4)
                h_prun_reorder_bin_inv = torch.logical_not(h_prun_reorder_bin)
                h_prun_reorder_bin_zero_bit_cnt = torch.sum(h_prun_reorder_bin_inv, dim=2)



                h_prun_top4k_sum = torch.round(128*torch.sum(h_prun_reorder_bin_zero_bit_cnt[0:256, :], dim=0)/256).int()

                final = torch.max(och32_shuffled_list,h_prun_top4k_sum)

                print(l,"th layer och32 input feeding                    :  ", h_prun_top4k_sum, " abs: ", torch.sum(h_prun_top4k_sum))
                print(l,"th layer och32 shuffle+input feeding            :  ", final, " abs: ", torch.sum(final))
                print(l,"th layer och32 shuffle+input feeding performance:  ", torch.sum(final-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                print("Batch = ", h_prun_reorder.shape[0], ", ICH = ", h_prun_reorder.shape[1], ", OCH = ", 128, ", Total Weight Zero Bit = ", h_prun_reorder.shape[1]*128*4)
                #### Inference ####
                h = self.sigma_net[l](h)
                h_prun = self.sigma_pruned_net[l](h_prun)
            else:

                # Original
                h = self.sigma_net[l](h)
                
                # Pruning using batch group
                bg_num = int(h_batch/self.bg_size)

                for bg in range (bg_num):
                    w_cur_layer = self.sigma_net[l].weight

                    w_pruned_cur_layer = w_cur_layer * w_nxt_masks[bg]
                  
                    width = w_cur_layer[w_nxt_masks[bg]].shape[0]
                    height = w_cur_layer[w_nxt_masks[bg]].shape[1]
                    weight_zero = (w_cur_layer[w_nxt_masks[bg]] == 0).sum()/width/height
                    weight_25 = torch.logical_and( (w_cur_layer[w_nxt_masks[bg]] < 2**-1), (w_cur_layer[w_nxt_masks[bg]] > 2**-4))
                    weight_25_count = (weight_25 == 1).sum()/width/height
                    
                    
                    self.sigma_pruned_net[l].weight.data = w_pruned_cur_layer

                    ##### Weight Reordering #####
                    w_cur_layer = w_cur_layer.T
                    weight = w_cur_layer[w_nxt_masks[bg]]

                    weight_ich_prun = torch.sum(weight==0, dim=1)
                    weight_remain_ich = weight_ich_prun != weight.shape[1]
                    weight = weight[weight_remain_ich]

                    weight_not_zero = weight[weight!=0]
                    weight_smallest_unit = torch.min(torch.abs(weight_not_zero))
                    weight_q = weight/weight_smallest_unit
                    weight_bin = dec2bin(weight_q.int(), 4)
                    weight_inv = torch.logical_not(weight_bin)
                    weight_zero_bit_cnt = torch.sum(weight_inv, dim=2)
                    weight_zero_bit_ich = torch.sum(weight_zero_bit_cnt, dim=1)

                    weight_zero_bit_cnt_sort = weight_zero_bit_cnt.clone().detach()
                    weight_bin_dup = weight_bin.clone().detach()
                    
                    sort_val, sort_idx = torch.sort(weight_zero_bit_ich, descending=True)

                    if sort_idx.shape[0]-int(sort_idx.shape[0]/4)*4 != 0:
                        group_num = int(sort_idx.shape[0]/4)+1
                    else:
                        group_num = int(sort_idx.shape[0]/4)

                    top_4k = sort_idx[0:group_num*2]

                    top_4k_slice = top_4k[0:group_num]
                    zero_max_list = weight_zero_bit_ich[top_4k_slice]

                    #weight_bin_dup = delete_row_tensor(weight_bin_dup, top_4k_slice)



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



                    if sort_idx.shape[0]-int(sort_idx.shape[0]/4)*4 != 0:
                        weight_loop = int(sort_idx.shape[0]/4)+1
                    else:
                        weight_loop = int(sort_idx.shape[0]/4)


                    
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

                    ##########
                    print(l,"th layer zero_max idx (before shuffle):            ", non_shuffled_zero, " abs: ", torch.sum(non_shuffled_zero))
                    print(l,"th layer zero_max idx (after och 128 shuffle):     ", zero_max_list, " abs: ", torch.sum(zero_max_list))
                    print(l,"th layer zero_max idx (after och 32 shuffle):      ", och32_shuffled_list, " abs: ", torch.sum(och32_shuffled_list))
                    print(l,"th layer zero_max idx och 128 shuffle performance: ", torch.sum(zero_max_list-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                    print(l,"th layer zero_max idx och 32 shuffle performance:  ", torch.sum(och32_shuffled_list-non_shuffled_zero)/torch.sum(non_shuffled_zero))

                    h_prun_slice = h_prun[bg*self.bg_size:bg*self.bg_size+self.bg_size]

                    #### Input Reordering ####
                    h_prun_reorder = h_prun_slice[:, top_4k_slice]
                    h_prun_reorder_not_zero = h_prun_reorder[h_prun_reorder != 0] 
                    input_smallest_unit = torch.min(torch.abs(h_prun_reorder_not_zero))
                    input_q = h_prun_reorder/input_smallest_unit
                    h_prun_reorder_bin = dec2bin(input_q.int(), 4)
                    h_prun_reorder_bin_inv = torch.logical_not(h_prun_reorder_bin)
                    h_prun_reorder_bin_zero_bit_cnt = torch.sum(h_prun_reorder_bin_inv, dim=2)
                    h_prun_top4k_sum = torch.round(128*torch.sum(h_prun_reorder_bin_zero_bit_cnt, dim=0)/32).int()

                    final = torch.max(och32_shuffled_list,h_prun_top4k_sum)
                    print(l,"th layer och32 input feeding                    :  ", h_prun_top4k_sum, " abs: ", torch.sum(h_prun_top4k_sum))
                    print(l,"th layer och32 shuffle+input feeding            :  ", final, " abs: ", torch.sum(final))
                    print(l,"th layer och32 shuffle+input feeding performance:  ", torch.sum(final-non_shuffled_zero)/torch.sum(non_shuffled_zero))
                    print("Batch = ", h_prun_reorder.shape[0], ", ICH = ", h_prun_reorder.shape[1], ", OCH = ", 128, ", Total Weight Zero Bit = ", h_prun_reorder.shape[1]*128*4)

                    #pdb.set_trace()
                    
                    #### Inference
                    h_slice = self.sigma_pruned_net[l](h_prun_slice)

                    if bg == 0:
                        h_slice_cat = h_slice
                    else:
                        h_slice_cat = torch.cat([h_slice_cat, h_slice], dim=0)                

                h_prun = h_slice_cat




                #pdb.set_trace()

            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
                h_prun = F.relu(h_prun, inplace=True)
            

                #input_smallest_unit =torch.min(torch.abs(h_prun[h_prun!=0])) 
                #input_ = h_prun/input_smallest_unit
                #input_bin = dec2bin(input_.int(), 4)
                #input_zero_bit_cnt = torch.sum(torch.logical_not(input_bin), dim=2)

                #zero_input_count = 0
                #
                #for batch in range (0, h_prun.shape[0]):
                #    sort_val, sort_idx = torch.sort(h_prun[batch,:], descending=True)

                #for ich in range (0, h_prun.shape[1], 4):
                #    zero_input_count = torch.sum(h_prun[:, ich] == 0)/h_prun.shape[0]

                #    if ich == 0:
                #        zero_input_list = zero_input_count.reshape(1)
                #    else:
                #        zero_input_list = torch.cat([zero_input_list, zero_input_count.reshape(1)], dim=0)

                #print(l, "th layer 4k ich input sparsity: ",zero_input_list)

                #for ich in range (0, h_prun.shape[1], 4):
                #    input_zero_bit_cnt[input_zero_bit_cnt == 0]


            #### SR batch group pruning ratio ####
            bg_num = int(h_batch / self.bg_size)


            w_nxt_masks = []

            if l == 4:
                p_ratio = 0.2
            else:
                p_ratio = 0.2

            for bg in range (bg_num):
                h_prun_slice = h_prun[bg*self.bg_size:bg*self.bg_size+self.bg_size]
                h_prun_slice_mean = h_prun_slice.mean(0)
                #h_prun_slice_mean_bool = h_prun_slice_mean != 0

                h_prun_slice_mean_max = h_prun_slice_mean.max()
                h_prun_slice_th = h_prun_slice_mean_max * p_ratio
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
        sigma_prun = trunc_exp(h_prun[..., 0])

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

        print(sigma.shape)
        print(sigma)

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
        pdb.set_trace()

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'sigma_prun': sigma_prun
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

def delete_row_tensor(a, del_row):
    device = a.device
    n = a.cpu().detach().numpy()
    del_row = del_row.cpu().detach().numpy()
    n = np.delete(n, del_row, 0)

    n = torch.from_numpy(n).to(device)
    return n
