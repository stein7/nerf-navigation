import torch
import pdb


def alloc_switching( wbin, xbin ):
    # ich, och, bit_width
    dim = len(wbin.shape)
    print(f'och: {wbin.shape[1]} --> I-Rounter Total: {int(wbin.shape[1]/16)}') 
    # cnt bit sparsity (ich, och)
    bit_sp = (1-wbin).sum(dim=dim-1)

    bit_sp_T = bit_sp.T

    # 4*32 unit bit sparsity
    w_unit = bit_sp_T.reshape(-1, 16, bit_sp.shape[0]).transpose(1,2)

    # Accum och direc bit sparsity (och division(w_unit num), ich)
    w_och = w_unit.sum(dim=dim-1)

    # descending; w ordering
    w_och_descend, descend_idx = torch.sort( w_och, descending=True, dim=1 )

    # bit sparsity using weight alloc (pe arr num, 4k row num) 
    w_alloc = w_och_descend[:, 0:int(w_och_descend.shape[1]/4)]
    shuffle_ich_idx = descend_idx[:,0:int(w_och_descend.shape[1]/4)]
    
    w_origin = w_och[:,0:int(w_och_descend.shape[1]/4)]

    if xbin.shape[2] %2 != 0 : 
        zero = torch.zeros( [ xbin.shape[0], xbin.shape[1], 1 ], device="cuda" )
        x = torch.cat([zero, xbin], dim=2)
        lsb_slice = x[:,:,0:4]
        msb_slice = x[:,:,4:8]
        lsb_slice_sp = (1-lsb_slice).sum(dim=2)
        msb_slice_sp = (1-msb_slice).sum(dim=2)

    xbin_sp = 1-xbin
    xbin_sp = xbin_sp.sum(dim=2)
    for pe in range( int(w_alloc.shape[0]) ):
        x_4k = xbin[:,shuffle_ich_idx[pe],:]
        x_4k_origin = xbin[:,0:shuffle_ich_idx[pe].shape[0],:]
        
        w_4k = w_alloc[pe].reshape(1,-1,1)
        w_4k_origin = w_origin[pe].reshape(1,-1,1)
        

        xbin_order = xbin_sp[:, shuffle_ich_idx[pe]]
        w_alloc_1 = wbin.shape[2]*16-w_alloc[pe]
        switch = xbin_order*w_alloc_1

       
        bs_w_alloc = x_4k*w_4k
        bs_origin = x_4k_origin * w_4k_origin
        bs_switching = bs_w_alloc.sum(dim=2).clone()
        T = x_4k_origin.shape[0] * x_4k_origin.shape[1] * x_4k_origin.shape[2] * wbin.shape[2] * 16

        sw_ctrlr = (1-x_4k).sum(dim=2)
        ibs_en = sw_ctrlr >= xbin.shape[2]/2

        bs_switching[ibs_en] = switch[ibs_en]
        
        max_sp = wbin.shape[2] * 16
        
        default_bit_sparsity_origin = (bs_origin == 0).sum()*max_sp
        default_bit_sparsity_w_alloc = (bs_w_alloc == 0).sum() * max_sp

        w_alloc_sp = bs_w_alloc.sum()
        origin_sp = bs_origin.sum()
        
        print(f'===== I-Router {pe} =====')
        print(f'origin : {(origin_sp+default_bit_sparsity_origin)/T}')
        print(f'w alloc: {(w_alloc_sp+default_bit_sparsity_w_alloc)/T}')
        print(f'switch: {(bs_switching.sum()+default_bit_sparsity_w_alloc)/T}')
    
    del w_unit
    del w_och
    del w_och_descend
    del descend_idx
    del xbin_sp
    del xbin_order
    del switch
    del bs_w_alloc
    del bs_origin
    del bs_switching
    del default_bit_sparsity_w_alloc
    del default_bit_sparsity_origin

