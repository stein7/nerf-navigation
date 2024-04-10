import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

from lib.approx import *
from lib.writer import *

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
                 opt=None,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []

        self.opt = opt
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
           
            if not opt.test         : sigma_net.append( CFC( in_features=in_dim, out_features=out_dim, bias=False, training=True , inf_accurate=True, eval_ich_range=opt.eval_ich_range) )
            elif opt.eval_accurate  : sigma_net.append( CFC( in_features=in_dim, out_features=out_dim, bias=False, training=False, inf_accurate=opt.eval_accurate, eval_ich_range=opt.eval_ich_range) )
            elif opt.eval_approx    : sigma_net.append( CFC( in_features=in_dim, out_features=out_dim, bias=False, training=False ) )
            else                    : sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

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
            
            #color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if not opt.compressor: color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            else: color_net.append( CFC( in_features=in_dim, out_features=out_dim, bias=False, training=True , inf_accurate=True) )

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
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

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
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_accurate =True

            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat
        }

    def sr_ich_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_quant=self.opt.inf_quant
            self.sigma_net[l].bp_quant=self.opt.bp_quant

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        
        return sigma



    def sr_accurate_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_accurate =True
            self.sigma_net[l].inf_msb = False
            self.sigma_net[l].inf_approx = False

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        
        return sigma

    def sr_approx_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_approx=True
            self.sigma_net[l].inf_accurate =False
            self.sigma_net[l].inf_msb = False

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        
        return sigma



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

    def sr_msb_compen_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_approx=False
            self.sigma_net[l].inf_accurate =False
            self.sigma_net[l].inf_msb = True

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return sigma

    def sr_switching_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_approx=False
            self.sigma_net[l].inf_accurate =False
            self.sigma_net[l].inf_msb = False
            self.sigma_net[l].inf_switching = True
            self.sigma_net[l].inf_w_alloc = False

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]        
        return sigma

    def sr_w_alloc_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_approx=False
            self.sigma_net[l].inf_accurate =False
            self.sigma_net[l].inf_msb = False
            self.sigma_net[l].inf_switching = False
            self.sigma_net[l].inf_w_alloc = True

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]        
        return sigma

    def sr_layer_wise_mode_v0_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_approx=False
            self.sigma_net[l].inf_accurate =False
            self.sigma_net[l].inf_msb = False
            self.sigma_net[l].inf_switching = False
            self.sigma_net[l].inf_w_alloc = False
            self.sigma_net[l].layer_wise_m0 = True
            self.sigma_net[l].msb_w_alloc_switch = False

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]        
        return sigma

    def sr_msb_switch_debug(self, x):
        # x: [N, 3], in [-bound, bound]
        

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            
            self.sigma_net[l].layer = l
            self.sigma_net[l].inf_approx=False
            self.sigma_net[l].inf_accurate =False
            self.sigma_net[l].inf_msb = False
            self.sigma_net[l].inf_switching = False
            self.sigma_net[l].inf_w_alloc = False
            self.sigma_net[l].layer_wise_m0 = False
            self.sigma_net[l].msb_w_alloc_switch = True

            h = self.sigma_net[l](h)
            
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]        
        return sigma

    def sr_accurate_err_debug(self, file_name, x): 
        #for l in range(self.num_layers):
        #    name = file_name+str(l)+'.csv'
        #    layer_err = self.sigma_net[l].accurate_err[0]
        #    np.savetxt(file_name+str(l)+'.csv', layer_err.detach().cpu().numpy(), delimiter=",")
        #    print(f'save csv file @ {name}')
        name = file_name+'_grad.csv'
        print(f'save csv file @ {name}')
        np.savetxt(file_name+'_grad.csv', x.grad.detach().cpu().numpy(), delimiter=",")

    def sr_approx_err_debug(self, file_name, x): 
        #for l in range(self.num_layers):
        #    name = file_name+str(l)+'.csv'
        #    layer_err = self.sigma_net[l].approx_err[0]
        #    np.savetxt(file_name+str(l)+'.csv', layer_err.detach().cpu().numpy(), delimiter=",")
        #    print(f'save csv file @ {name}')
        name = file_name+'_grad.csv'
        print(f'save csv file @ {name}')
        np.savetxt(file_name+'_grad.csv', x.grad.detach().cpu().numpy(), delimiter=",")

    def sr_layer_wise_mode_v0_err_debug(self, file_name, x): 
        name = file_name+'_grad.csv'
        print(f'save csv file @ {name}')
        np.savetxt(file_name+'_grad.csv', x.grad.detach().cpu().numpy(), delimiter=",")

    def sr_msb_compen_err_debug(self, file_name, x): 
        name = file_name+'_grad.csv'
        print(f'save csv file @ {name}')
        np.savetxt(file_name+'_grad.csv', x.grad.detach().cpu().numpy(), delimiter=",")

    def sr_msb_switch_err_debug(self, file_name, x): 
        name = file_name+'_grad.csv'
        print(f'save csv file @ {name}')
        np.savetxt(file_name+'_grad.csv', x.grad.detach().cpu().numpy(), delimiter=",")


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
