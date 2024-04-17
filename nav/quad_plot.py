import torch
from torch._C import device
import numpy as np
import json

from .math_utils import rot_matrix_to_vec
from .quad_helpers import astar, next_rotation

import pdb
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as pyo
import plotly.io as pio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Planner:
    def __init__(self, start_state, end_state, cfg, density, density_fn, density_prun_fn, opt):
        self.nerf = density_fn
        self.nerf_prun = density_prun_fn
        self.nerf_origin = density
        self.opt = opt

        self.cfg                = cfg
        self.T_final            = cfg['T_final']
        self.steps              = cfg['steps']
        self.lr                 = cfg['lr']
        self.epochs_init        = cfg['epochs_init']
        self.epochs_update      = cfg['epochs_update']
        self.fade_out_epoch     = cfg['fade_out_epoch']
        self.fade_out_sharpness = cfg['fade_out_sharpness']
        self.mass               = cfg['mass']
        self.J                  = cfg['I']
        self.g                  = torch.tensor([0., 0., -cfg['g']])
        self.body_extent        = cfg['body']
        self.body_nbins         = cfg['nbins']

        self.CHURCH = False

        self.dt = self.T_final / self.steps

        self.start_state = start_state
        self.end_state   = end_state

        slider = torch.linspace(0, 1, self.steps)[1:-1, None]

        states = (1-slider) * self.full_to_reduced_state(start_state) + \
                    slider  * self.full_to_reduced_state(end_state)

        self.states = states.clone().detach().requires_grad_(True)
        self.initial_accel = torch.tensor([cfg['g'], cfg['g']]).requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        #self.body_extent = torch.tensor([[-0.0005, 0.0005], [-0.0005, 0.0005], [-0.0002, 0.0002]])
        #self.body_nbins = torch.tensor([10, 10, 5])

        body = torch.stack( torch.meshgrid( torch.linspace(self.body_extent[0, 0], self.body_extent[0, 1], self.body_nbins[0]),
                                            torch.linspace(self.body_extent[1, 0], self.body_extent[1, 1], self.body_nbins[1]),
                                            torch.linspace(self.body_extent[2, 0], self.body_extent[2, 1], self.body_nbins[2])), dim=-1)
        self.robot_body = body.reshape(-1, 3)

        if self.CHURCH:
            self.robot_body = self.robot_body/2

        self.epoch = 0

        self.init_loss_log = []


    def full_to_reduced_state(self, state):
        pos = state[:3]
        R = state[6:15].reshape((3,3))

        x,y,_ = R @ torch.tensor( [1.0, 0, 0 ] )
        angle = torch.atan2(y, x)

        return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()

    def a_star_init(self):
        side = 100 #PARAM grid size
        #side = 10

        if self.CHURCH:
            x_linspace = torch.linspace(-2,-1, side)
            y_linspace = torch.linspace(-1,0, side)
            z_linspace = torch.linspace(0,1, side)

            coods = torch.stack( torch.meshgrid( x_linspace, y_linspace, z_linspace ), dim=-1)
        else:
            linspace = torch.linspace(-1,1, side) #PARAM extends of the thing
            # side, side, side, 3
            coods = torch.stack( torch.meshgrid( linspace, linspace, linspace ), dim=-1)

        kernel_size = 5 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size
       
        #pdb.set_trace()
        #coods_2d = coods.reshape(-1, 3)
        #coods_2d = coods_2d.clone().detach()
        #coods_2d.requires_grad_(True)

        outputs = self.nerf_origin(coods)
        output = outputs['sigma'].reshape(coods.shape[:-1])                         # batch grouping / input reordering
        
        '''
        # drawing histogram 
        density_data = output.cpu().flatten()
        density_occ = density_data[torch.argwhere(density_data>0.3)][...,0]
        density_log = torch.log10(density_occ)
        #density_data = np.random.randint(0, 100, size=(100,100,100))
        #density_data = density_data.flatten()
        bins = np.arange(0,11.5,0.5)#(0, density_data.max(), 1e+8)
        hist, bin_edges = np.histogram(density_log, bins)
        plt.figure(figsize=(10,6))
        plt.hist(density_log, bins=bins, edgecolor='k')
        plt.title('Distribution of density (occupied, logscale)')
        plt.xlabel('density value (log scale)')
        plt.ylabel('distribution')
        plt.xticks(bin_edges)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('density_histogram.png')
        pdb.set_trace()
        '''
        
        #pdb.set_trace()
        
        #output = outputs['sigma']

        #output.requires_grad_(True)
        #autograd_grad = torch.autograd.grad(output, coods_2d, grad_outputs=torch.ones_like((output), device='cuda'), retain_graph=True, create_graph=True, allow_unused=True)[0]
        #output.flatten().backward(torch.ones((coods_2d.shape[0]), device='cuda'), retain_graph=True)

        #bpgrad = coods.grad
        
        #print(torch.all(autograd_grad == bpgrad))
        #pdb.set_trace()

        '''
        data = output.cpu()
        data_oc = data > 0.3
        colors = plt.cm.viridis(data)
        #colors = np.zeros((5,5,5,4))
        
        #data_norm = data_/(data.max())
        #colors[:,:,:,-1] = data_norm
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(data_oc, facecolors=colors, edgecolor=None)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

        plt.savefig('res_sigma3d_occupied_colored.png')
        '''

        

        output_prun = outputs['sigma_prun'].reshape(coods.shape[:-1])


        #print("non-pruned output inference start")
        #output = self.nerf(coods)
        #print("non-pruned output inference done")
        #print("pruned output inference start")
        #output_prun = self.nerf_prun(coods)
        #print("pruned output inference done")


        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)
        
        # 20, 20, 20
        occupied = maxpool(output[None,None,...])[0,0,...] > 0.3
        occupied_prun = maxpool(output_prun[None,None,...])[0,0,...] > 0.3
        #pdb.set_trace()
        #pdb.set_trace()

        #### Save Occupancy Val as csv ####
        output_filename = './occupancy_log/a_star_output_log.csv'
        occupancy_filename = './occupancy_log/a_star_occupancy_log.csv'


        with open (occupancy_filename, 'w', newline='') as f:
           writer = csv.writer(f)
           writer.writerows(occupied.reshape(8,-1).cpu().numpy())
           torch.save(occupied, './occupancy_log/occupancy.pth')

        with open (output_filename, 'w', newline='') as f_out:
           writer = csv.writer(f_out)
           writer.writerows(output.reshape(10,-1).cpu().numpy())
           torch.save(output, './occupancy_log/output.pth')

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3]  + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3]  + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        print(start, end)

        path = astar(occupied, start, end)                                              # 21 elements in A*
        #-----------sungmin was here------------------
        path_tensor = torch.tensor(path, dtype=torch.float)
        def lin_interp(path, num):
            
            len_init = len(path)
            alpha_init = torch.linspace(0, 1, num+1)
            res = torch.zeros((len_init + (num-1)*(len_init-1), 3))

            for i in range(len_init - 1):
                alpha_inv, path_cur = torch.meshgrid(1 - alpha_init, path[i])
                alpha, path_next = torch.meshgrid(alpha_init, path[i+1])
                insert = (path_cur*alpha_inv + path_next*alpha)[:-1]
                res[i*num:i*num+num] = insert
                
            res[-1] = path[-1]
            return res

        interp_path = lin_interp(path_tensor, 10)  
        #---------------------------------------------
        # convert from index cooredinates
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1             # 21,3

        #adding way
        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)   # 21,4

        #prevents weird zero derivative issues
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        states = (prev_smooth + next_smooth + states)/3                                 # 21,4
        
        # with torch.no_grad(): #grad연산이 기록되었던 시점 주의
        #     states[11, :3] = torch.tensor([0.168, 0.2865, 0.2])
        #     states[12, :3] = torch.tensor([0.1, 0.27, 0.2]) # 0.27 {0.3} 0.32
        #     states[13, :3] = torch.tensor([0.064, 0.3334, 0.2])
        self.states = states.clone().detach().requires_grad_(True)
    
        print("End of A*")

    def dh_init(self, start, end, points):
        
        self.start_pos = start.cpu().numpy()
        self.end_pos = end.cpu().numpy()

        # self.start_pos = np.array(start)
        # self.end_pos = np.array([-0.6, -0.7, 0.2])
        num_points = points

        # 시작점에서 종료점까지의 벡터 계산
        delta = self.end_pos - self.start_pos

        # 시작점과 종료점을 제외한 중간 점 생성
        intermediate_points = [self.start_pos + i * delta / (num_points + 1) for i in range(1, num_points + 1)]

        # 결과를 float 텐서로 변환
        intermediate_points_tensor = np.array(intermediate_points, dtype=np.float32)
        intermediate_points_tensor = torch.tensor(intermediate_points_tensor, dtype=torch.float32)

        states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1)
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        self.states = states.clone().detach().requires_grad_(True)

    def params(self):
        return [self.initial_accel, self.states]

    def calc_everything(self):

        start_pos   = self.start_state[None, 0:3]                   # 18 len to 3
        start_v     = self.start_state[None, 3:6]
        start_R     = self.start_state[6:15].reshape((1, 3, 3))
        start_omega = self.start_state[None, 15:]

        end_pos   = self.end_state[None, 0:3]
        end_v     = self.end_state[None, 3:6]
        end_R     = self.end_state[6:15].reshape((1, 3, 3))
        end_omega = self.end_state[None, 15:]

        next_R = next_rotation(start_R, start_omega, self.dt)

        # start, next, decision_states, last, end

        start_accel = start_R @ torch.tensor([0,0,1.0]) * self.initial_accel[0] + self.g
        next_accel = next_R @ torch.tensor([0,0,1.0]) * self.initial_accel[1] + self.g

        next_vel = start_v + start_accel * self.dt
        after_next_vel = next_vel + next_accel * self.dt

        next_pos = start_pos + start_v * self.dt
        after_next_pos = next_pos + next_vel * self.dt
        after2_next_pos = after_next_pos + after_next_vel * self.dt
    
        # position 2 and 3 are unused - but the atached roations are
        current_pos = torch.cat( [start_pos, next_pos, after_next_pos, after2_next_pos, self.states[2:, :3], end_pos], dim=0)           #24, 3

        current_pos_dh = torch.cat( [start_pos, start_pos, self.states[:, :3], end_pos], dim=0)

        prev_pos = current_pos[:-1, :]
        next_pos = current_pos[1: , :]

        current_vel = (next_pos - prev_pos)/self.dt
        current_vel = torch.cat( [ current_vel, end_v], dim=0)

        prev_vel = current_vel[:-1, :]
        next_vel = current_vel[1: , :]

        current_accel = (next_vel - prev_vel)/self.dt - self.g

        # duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel = torch.cat( [ current_accel, current_accel[-1,None,:] ], dim=0)

        accel_mag     = torch.norm(current_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = current_accel/accel_mag

        # remove states with rotations already constrained
        z_axis_body = z_axis_body[2:-1, :]

        z_angle = self.states[:,3]

        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        rot_matrix = torch.cat( [start_R, next_R, rot_matrix, end_R], dim=0)

        current_omega = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt
        current_omega = torch.cat( [ current_omega, end_omega], dim=0)

        prev_omega = current_omega[:-1, :]
        next_omega = current_omega[1:, :]

        angular_accel = (next_omega - prev_omega)/self.dt
        # duplicate last ang_accceleration - its not actaully used for anything (there is no action at last state)
        angular_accel = torch.cat( [ angular_accel, angular_accel[-1,None,:] ], dim=0)

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[...,None])[...,0]
        actions =  torch.cat([ accel_mag*self.mass, torques ], dim=-1)

        return current_pos_dh, current_vel, current_accel, rot_matrix, current_omega, angular_accel, actions

    def get_full_states(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def get_actions(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        if not torch.allclose( actions[:2, 0], self.initial_accel ):
            print(actions)
            print(self.initial_accel)
        return actions

    def get_next_action(self):
        actions = self.get_actions()
        # fz, tx, ty, tz
        return actions[0, :]

    def body_to_world(self, points):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        
        num_matrices = self.states.size(0) + 3
        identity_matrices = torch.eye(3).unsqueeze(0).expand(num_matrices, -1, -1)  

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        # world_points =  rot_matrix @ points.T + pos[..., None]
        world_points =  identity_matrices @ points.T + pos[..., None]

        return world_points.swapdims(-1,-2)

    def get_state_cost(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0].to(device)
        torques = torch.norm(actions[:, 1:], dim=-1).to(device)

        # S, B, 3  =  S, _, 3 +      _, B, 3   X    S, _,  3
        B_body, B_omega = torch.broadcast_tensors(self.robot_body, omega[:,None,:])         # 24, 500, 3(x,y,z)
        point_vels = vel[:,None,:] + torch.cross(B_body, B_omega, dim=-1)

        # S, B
        distance = torch.sum( vel**2 + 1e-5, dim = -1)**0.5
        # S, B
        density = self.nerf( self.body_to_world(self.robot_body) )**2

        ################-------- DH -------############################
        # pdb.set_trace()
        print("#### data processing start ####")

        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)
        density = self.nerf( self.body_to_world(self.robot_body) )**2 # (24, 500)
        colision_prob = torch.mean(density, dim = -1)  # (24, 1)
        loss = torch.mean(colision_prob) # (1, 1)
        loss.backward()


        # path coord (21+3,3)
        self.save_poses(self.basefolder / "init_poses" / ("start.json"))
        # body points (24, 500, 3)
        coods = self.body_to_world(self.robot_body)
        # density points (24, 500)
        sigma = self.nerf(coods)**2 
        # body BP grad (24, 500, 3)
        coods_BP = torch.autograd.grad(sigma, coods, grad_outputs=torch.ones_like((sigma),device='cuda'), retain_graph=True, create_graph=True, allow_unused=True)[0]
        
        
        # device, array
        coods = coods.to('cpu')
        coods_array = coods.detach().numpy()
        sigma = sigma.to('cpu')
        coods_BP = coods_BP.to('cpu')
        coods_BP_array = coods_BP.detach().numpy()
        
        # Path info Summary
        num_ways = coods.shape[0]
        main_ways = torch.where( torch.mean(sigma, dim=-1) > 0.3 )[0].tolist()
        print(f"Way points : { num_ways }")
        print(f"Collision Way points : { main_ways } ")


        # pdb.set_trace()
        print("Projection & edge detection & n vec & cos sim / ratio")
        #--------------------------Projection & edge detection & n vec & cos sim / ratio--------------------------#
        threshold = 0.3
        coods_z_proj, sigma_z_proj, sigma_z_proj_value = [], [], []
        for Layer in range(0, 5): # Layer = 4 # z축 기준 0~4층, index사용법: coods_z_proj[Layer][Way]
            coods_z_proj.append([ torch.stack([ coods[way][Layer+i, :] for i in range(0, 500, 5) ])               for way in range(0, num_ways) ])
            sigma_z_proj.append([ torch.stack([ sigma[way][Layer+i]    for i in range(0, 500, 5) ]) > threshold   for way in range(0, num_ways) ])
            sigma_z_proj_value.append([ torch.stack([ sigma[way][Layer+i]    for i in range(0, 500, 5) ])         for way in range(0, num_ways) ])
        
        # All Layer z axis projection
        sigma_z_proj_all = [ sigma_z_proj[4][way] | sigma_z_proj[3][way] | sigma_z_proj[2][way] | sigma_z_proj[1][way] | sigma_z_proj[0][way]  for way in range(0, num_ways) ]
        
        # Layer별 occupied bitmap과 value 분포 ratio 출력
        for L in range(4,-1, -1):
            way = self.opt.waypoint
            print(f"Way {way} Layer {L} Sigma Values")
            for i, value in enumerate(sigma_z_proj_value[L][way], start=1):
                print(f'{value:.4e}', end=', ' if i % 10 != 0 else '\n')  
        for L in range(4,-1, -1):
            way = self.opt.waypoint
            print(f"Way {way} Layer {L} occupied bitmap | occupied ratio : { len( torch.where(sigma_z_proj[L][way] == True)[0] ) }%")
            print(sigma_z_proj[L][way])


        # ideal edge detection : sobel filter -> True idx detecting, Problems : 1) high resolution -> many edges. 2) idx arrange문제 -> 선 형태로 추출안됨  
        sigma_edge_idx = [ torch.where( sigma_z_proj_all[way] )[0] for way in range(0, num_ways)]
        
        # Way, Layer별로 nvec와의 cos sim 값을 출력 (input : Layer, Way, edge idx)
        for Layer in range(4,-1, -1):
            # manual edge detection (Layer별)
            """ ###-- Stonehenge Path1 
            if Layer == 0 : sigma_edge_idx[13] = torch.tensor([40, 41, 52, 53, 54, 35, 26, 16, 6]) # L0
            if Layer == 1 : sigma_edge_idx[13] = torch.tensor([40, 41, 52, 53, 54, 35, 26, 16, 6]) # L1
            if Layer == 2 : sigma_edge_idx[13] = torch.tensor([40, 41, 52, 53, 54, 35, 16, 6]) # L2
            if Layer == 3 : sigma_edge_idx[13] = torch.tensor([30, 41, 62, 53, 54, 45, 16, 7]) # L3
            if Layer == 4 : sigma_edge_idx[13] = torch.tensor([30, 41, 52, 53, 54, 45, 16, 7]) # L4

            if Layer == 0 : sigma_edge_idx[14] = torch.tensor([10, 11, 12, 13, 24, 34, 44, 54, 63, 73, 83, 92]) # L0
            if Layer == 1 : sigma_edge_idx[14] = torch.tensor([10, 11, 12, 13, 25, 54, 83, 92]) # L1
            if Layer == 2 : sigma_edge_idx[14] = torch.tensor([10, 11, 12, 13, 25, 34, 44, 54, 73, 92]) # L2
            if Layer == 3 : sigma_edge_idx[14] = torch.tensor([10, 11, 12, 13, 25, 34, 44, 54, 73, 92]) # L3
            if Layer == 4 : sigma_edge_idx[14] = torch.tensor([1, 13, 24, 34, 44, 54, 63, 73, 82, 92]) # L4

            if Layer == 0 : sigma_edge_idx[15] = torch.tensor([3, 23, 22, 11, 10, 70, 81, 82, 91]) # L0
            if Layer == 1 : sigma_edge_idx[15] = torch.tensor([3, 23, 22, 21, 10, 70, 81, 82, 91]) # L1
            if Layer == 2 : sigma_edge_idx[15] = torch.tensor([3, 13, 22, 21, 10, 70, 81, 82, 91]) # L2
            if Layer == 3 : sigma_edge_idx[15] = torch.tensor([3, 13, 22, 11, 10, 70, 81, 82, 91]) # L3
            if Layer == 4 : sigma_edge_idx[15] = torch.tensor([3, 13, 12, 11, 10, 70, 81, 92]) # L4
            """
            """ ###-- Stonehenge Path2 
            if Layer == 0 : sigma_edge_idx[8] = torch.tensor([4, 15, 25, 36, 46, 56, 66, 77, 88, 98]) # L0
            if Layer == 1 : sigma_edge_idx[8] = torch.tensor([5, 15, 26, 36, 46, 57, 67, 77, 88, 98]) # L1
            if Layer == 2 : sigma_edge_idx[8] = torch.tensor([5, 15, 26, 36, 46, 57, 67, 68, 79]) # L2
            if Layer == 3 : sigma_edge_idx[8] = torch.tensor([5, 15, 26, 36, 47, 58, 59]) # L3
            if Layer == 4 : sigma_edge_idx[8] = torch.tensor([6, 16, 27, 38, 39]) # L4

            if Layer == 0 : sigma_edge_idx[9] = torch.tensor([20, 31, 41, 51, 61, 71, 82, 92, 93, 94, 95, 96, 97, 98, 99]) # L0
            if Layer == 1 : sigma_edge_idx[9] = torch.tensor([20, 31, 41, 51, 61, 72, 82, 93, 94, 95, 96, 97, 98, 89]) # L1
            if Layer == 2 : sigma_edge_idx[9] = torch.tensor([20, 31, 41, 52, 62, 63, 74, 65, 66, 57, 48, 49]) # L2
            if Layer == 3 : sigma_edge_idx[9] = torch.tensor([20, 31, 42, 43, 44, 34, 25, 26, 16, 7]) # L3
            if Layer == 4 : sigma_edge_idx[9] = torch.tensor([0, 11, 22, 43, 34, 24, 14, 6]) # L4

            if Layer == 0 : sigma_edge_idx[4] = torch.tensor([50, 61, 62, 63, 64, 65, 76, 67, 58, 69]) # L0
            if Layer == 1 : sigma_edge_idx[4] = torch.tensor([50, 51, 52, 53, 54, 65, 56, 57, 58, 69]) # L1
            if Layer == 2 : sigma_edge_idx[4] = torch.tensor([60, 51, 52, 63, 64, 65, 56, 57, 58, 59]) # L2
            if Layer == 3 : sigma_edge_idx[4] = torch.tensor([60, 61, 62, 63, 64, 65, 56, 57, 58, 59]) # L3
            if Layer == 4 : sigma_edge_idx[4] = torch.tensor([50, 61, 52, 63, 64, 65, 56, 57, 58, 59]) # L4
            """
            """ ###-- Stonehenge Path3
            if Layer == 0 : sigma_edge_idx[6] = torch.tensor([30, 31, 42, 43, 54, 64, 73, 83, 93]) # L0
            if Layer == 1 : sigma_edge_idx[6] = torch.tensor([40, 41, 42, 43, 54, 64, 74, 83, 93]) # L1
            if Layer == 2 : sigma_edge_idx[6] = torch.tensor([30, 42, 43, 54, 64, 73, 83, 93]) # L2
            if Layer == 3 : sigma_edge_idx[6] = torch.tensor([30, 42, 43, 54, 64, 73, 83, 93]) # L3
            if Layer == 4 : sigma_edge_idx[6] = torch.tensor([30, 31, 42, 43, 54, 64, 74, 83, 93]) # L4

            if Layer == 0 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 38, 39, 89, 88, 87, 86, 95]) # L0
            if Layer == 1 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 38, 39, 89, 88, 87, 86, 95]) # L1
            if Layer == 2 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 38, 39, 89, 88, 87, 86, 95]) # L2
            if Layer == 3 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 28, 39, 89, 88, 87, 86, 95]) # L3
            if Layer == 4 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 28, 29, 89, 88, 87, 86, 95]) # L4

            if Layer == 0 : sigma_edge_idx[28] = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 94]) # L0
            if Layer == 1 : sigma_edge_idx[28] = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 94]) # L1
            if Layer == 2 : sigma_edge_idx[28] = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]) # L2
            if Layer == 3 : sigma_edge_idx[28] = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]) # L3
            if Layer == 4 : sigma_edge_idx[28] = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]) # L4
            """
            
            """ ###-- Scannet Path1 -> th0.3 <-> th10 동시
            sigma_edge_idx[3] = torch.tensor([20, 31, 32, 43, 54, 65, 76, 87, 88, 89]) # L0 1 2 3 4 th10
            sigma_edge_idx[4] = torch.tensor([40, 31, 32, 33, 24, 25, 36, 45, 55, 75, 86, 96]) # L0 1 2 3 4 th10
            
            sigma_edge_idx[3] = torch.tensor([30, 41, 42, 43, 54, 64, 75, 85, 86, 87, 88, 89]) # L0 1 2 3 4 th0.3
            sigma_edge_idx[4] = torch.tensor([30, 31, 22, 23, 24, 25, 36, 46, 55, 65, 76, 86, 96]) # L0 1 2 3 4 th0.3
            sigma_edge_idx[12] = torch.tensor([6, 16, 26, 36, 46, 56, 66, 76, 86, 96]) # L0 1 2 3 4 th0.3
            """
            
            """ ###-- Replica Path1 -> th10으로 변경
            ## sigma_edge_idx[7] = torch.tensor([30, 31, 32, 33, 34, 24, 14, 4]) # L01234 - th > 10
            if Layer == 0 : sigma_edge_idx[7] = torch.tensor([40, 41, 42, 53, 54, 65, 55, 45, 35, 25, 16, 6, 8, 18, 29, 39, 48, 57, 67, 77, 87, 97]) # L0
            if Layer == 1 : sigma_edge_idx[7] = torch.tensor([60, 51, 52, 53, 64, 75, 65, 55, 45, 35, 25, 15, 5, 7, 18, 29, 38, 48, 57, 67, 77, 87, 97]) # L1
            if Layer == 2 : sigma_edge_idx[7] = torch.tensor([40, 51, 52, 53, 54, 65, 55, 45, 35, 25, 15, 5, 7, 17, 27, 38, 48, 58, 68, 77, 87, 97]) # L2
            if Layer == 3 : sigma_edge_idx[7] = torch.tensor([40, 41, 42, 33, 34, 35, 25, 15, 5, 8, 17, 27, 38, 48, 58,67, 77, 87, 96]) # L3
            if Layer == 4 : sigma_edge_idx[7] = torch.tensor([30, 31, 32, 33, 34,24, 14, 4, 7, 18, 28, 38, 48, 58, 68, 78, 88, 98]) # L4

            if Layer == 0 : sigma_edge_idx[6] = torch.tensor([50, 51, 62, 63, 74, 84, 95]) # L0
            if Layer == 1 : sigma_edge_idx[6] = torch.tensor([60, 61, 62, 63, 64, 75, 85, 95]) # L1
            if Layer == 2 : sigma_edge_idx[6] = torch.tensor([60, 61, 62, 63, 64, 75, 85, 95]) # L2
            if Layer == 3 : sigma_edge_idx[6] = torch.tensor([70, 61, 62, 63, 74, 85, 95]) # L3
            if Layer == 4 : sigma_edge_idx[6] = torch.tensor([70, 61, 62, 63, 74, 84, 94]) # L4

            if Layer == 0 : sigma_edge_idx[5] = torch.tensor([50, 41, 31, 21, 22, 23, 24, 25, 36, 46, 56, 65, 75, 84, 94]) # L0
            if Layer == 1 : sigma_edge_idx[5] = torch.tensor([50, 41, 31, 32, 33, 34, 35, 45, 55, 65, 75, 84, 94, 93, 82, 71, 61, 50]) # L1
            if Layer == 2 : sigma_edge_idx[5] = torch.tensor([41, 32, 33, 34, 35, 45, 54, 64, 74, 73, 72, 61, 51, 41]) # L2
            if Layer == 3 : sigma_edge_idx[5] = torch.tensor([42, 52, 62, 63, 44, 33, 42]) # L3
            if Layer == 4 : sigma_edge_idx[5] = torch.tensor([42, 52]) # L4
            """
            
            if Layer == 0 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 38, 39, 89, 88, 87, 86, 95]) # L0
            if Layer == 1 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 38, 39, 89, 88, 87, 86, 95]) # L1
            if Layer == 2 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 38, 39, 89, 88, 87, 86, 95]) # L2
            if Layer == 3 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 28, 39, 89, 88, 87, 86, 95]) # L3
            if Layer == 4 : sigma_edge_idx[8] = torch.tensor([7, 17, 27, 28, 29, 89, 88, 87, 86, 95]) # L4

            for way in range(self.opt.waypoint, self.opt.waypoint+1): #input : Layer edge + Layer cord
                n_vec, midpoint = [], []
                # coods_z_proj_L4 = coods_z_proj[4] # Assign cood Layer 
                
                for i in range( len(sigma_edge_idx[way])-1 ):
                    grad_vec = coods_z_proj[Layer][way][sigma_edge_idx[way][i+1]] - coods_z_proj[Layer][way][sigma_edge_idx[way][i]]
                    n_vec.append( torch.tensor([ -grad_vec[1], grad_vec[0], grad_vec[2] ]) )   # 2D는 (-y,x) 3D는 cross(a,b)이지만 이경우는 z사영이므로
                    midpoint.append( torch.tensor( (coods_z_proj[Layer][way][sigma_edge_idx[way][i+1]] + coods_z_proj[Layer][way][sigma_edge_idx[way][i]]) / 2.0 , dtype=torch.float32, requires_grad=True).to('cuda') )
                
                n_vec = torch.stack( n_vec )
                midpoint = torch.stack( midpoint )
                mid_sigma = self.nerf(midpoint) ** 2
                midpoint_BP = torch.autograd.grad(mid_sigma.mean(), midpoint, create_graph=True)[0]

                midp_cosim = torch.nn.functional.cosine_similarity(n_vec, midpoint_BP, eps=1e-45)

                mask = ~torch.isnan(midp_cosim)
                midp_cosim = midp_cosim[mask]
                # midp_cosim = torch.cat( (midp_cosim[:4], midp_cosim[5:]) ) # Stonehenge Path1 - Way15 한정
                
                result = [ int(abs(value) * 100) for value in midp_cosim ]
                print(f"Layer:{Layer}, Way: {way}, mid csim: {result}, mean: {abs(midp_cosim).mean()*100:.3}")

            
            print("#-- Visualize n-vec & midp grad --#")
            #-- Visualize n-vec & midp grad --#
            midpoint = midpoint.cpu().detach().numpy()
            midpoint_BP = midpoint_BP.cpu().detach().numpy()
            n_vec = n_vec.cpu().detach().numpy()
            for i in range(len(midpoint)):
                midpoint_BP[i] = midpoint_BP[i] / np.linalg.norm(midpoint_BP[i]) * 0.03
                n_vec[i] = n_vec[i] / np.linalg.norm(n_vec[i]) * 0.03
            fig, ax1 = plt.subplots(1, 1, subplot_kw={"projection":"3d"})
            for i in range(len(midpoint)):
                ax1.scatter(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2], c='red', marker='o', alpha=0.5)
                ax1.quiver(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2],
                        midpoint_BP[i, 0], midpoint_BP[i, 1], midpoint_BP[i, 2], color='blue')
                ax1.quiver(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2],
                        n_vec[i, 0], n_vec[i, 1], n_vec[i, 2], color='green')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title(f"Layer{Layer}, way{way}")
            # plt.show() 는 pdb에서 수동으로 하거나 생략하기
        
        
        pdb.set_trace()
        print("#### data processing ####")
        #--------------------------scaling--------------------------#
        coods_BP_normalize = []
        coods_BP_array = coods_BP_array.astype(np.float64) # inf값 방지용
        for i in range(num_ways):
            normalized_row = []
            for v in coods_BP_array[i]:
                if (v == 0).all():
                    normalized_row.append([0, 0, 0])
                else:
                    normalized_row.append(v / np.linalg.norm(v))
            coods_BP_normalize.append(normalized_row)
        coods_BP_normalize = np.array(coods_BP_normalize)
        scale = 0.01
        coods_BP_normalize = coods_BP_normalize * scale
        #--------------------------Occupied & TopBot Vec 추출--------------------------#
        threshold = 0.3
        occupied_idx_sigma  =   [ torch.where(sigma[i] >= threshold)[0].to('cpu') for i in range(0, num_ways) ]
        occupied_coods      =   [ coods[i][occupied_idx_sigma[i]] for i in range(0, num_ways) ]
        #--------------------------Global(way) Top 20% bot 20% 뽑기--------------------------#
        top_percent = 0.2
        k = int(top_percent * 500)
        global_num_grad = []
        for i in range(0, num_ways):
            _, top_max = torch.topk(sigma[i], k)
            _, bot_min = torch.topk(-sigma[i], k)
            cood_top = coods[i][top_max]
            cood_bot = coods[i][bot_min]
            Top_mean = torch.mean(cood_top, dim=0)
            Bot_mean = torch.mean(cood_bot, dim=0)
            global_num_grad.append( Top_mean - Bot_mean )
        #--------------------------Local Top 20% bot 20% 뽑기--------------------------#
        top_percent = 0.2
        local_num_grad = []
        for i in range(0,num_ways):
            k = int(top_percent * sigma[i][occupied_idx_sigma[i]].numel())
            _, top_max = torch.topk(sigma[i][occupied_idx_sigma[i]], k)
            _, bot_min = torch.topk(-sigma[i][occupied_idx_sigma[i]], k)
            cood_top = occupied_coods[i][top_max]
            cood_bot = occupied_coods[i][bot_min]
            Top_mean = torch.mean(cood_top, dim=0)
            Bot_mean = torch.mean(cood_bot, dim=0)
            local_num_grad.append( Top_mean - Bot_mean ) 
        #--------------------------Cos Similarity 계산 및 출력--------------------------#
        cos_sim = [ torch.nn.functional.cosine_similarity(global_num_grad[i], coods_BP[i], eps=1e-45).detach().numpy() for i in range(0, num_ways) ]  
        cos_sim_occupied = [ torch.nn.functional.cosine_similarity(global_num_grad[i], coods_BP[i][occupied_idx_sigma[i]], eps=1e-45).detach().numpy() for i in range(0, num_ways) ]
        
        bound = [-1.1, -0.8, 0.8, 1.1]
        for i in range(0, num_ways):
            count_between = np.count_nonzero( (np.digitize(cos_sim[i], bound) == 1) | (np.digitize(cos_sim[i], bound) == 3) )
            total_count = len(cos_sim[i])
            ratio = "{:.2f}".format( (count_between / total_count) * 100 )
            print(f"body {i} +/- ratio : {ratio}, len : {total_count}")
        for i in range(0, num_ways):
            count_between = np.count_nonzero( (np.digitize(cos_sim_occupied[i], bound) == 1) | (np.digitize(cos_sim_occupied[i], bound) == 3) )
            total_count = len(cos_sim_occupied[i])
            ratio_occupied = "{:.2f}".format( total_count/500*100 )
            if total_count != 0 : ratio = "{:.2f}".format( (count_between / total_count) * 100 )
            else: ratio = 0
            print(f"occupied {i} +/- ratio : {ratio}, len : { total_count } ( { ratio_occupied } % )")

        pdb.set_trace()
        print("#### Visulize normal vector ####")
        #------------Visulize normal vector--------------#
        way = 28
        ng_cood = global_num_grad[way].detach().numpy()
        ng_vec = global_num_grad[way].detach().numpy() 
        ng_vec = ng_vec / np.linalg.norm(ng_vec)
        to_vec = self.states.grad[:, :3][way-2].cpu().detach().numpy()
        to_vec = to_vec / np.linalg.norm(to_vec)
        print(f'grad cos sim : { torch.nn.functional.cosine_similarity( torch.from_numpy(ng_vec[None,:]), torch.from_numpy(to_vec[None,:]) ) }')

        fig, ax1 = plt.subplots(1, 1, subplot_kw={"projection":"3d"})
        ax1.quiver(ng_cood[0], ng_cood[1], ng_cood[2],
                   ng_vec[0], ng_vec[1], ng_vec[2],
                   color='red')
        ax1.quiver(ng_cood[0], ng_cood[1], ng_cood[2],
                   to_vec[0], to_vec[1], to_vec[2], color='blue')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f"way{way}")
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        plt.show()
        #--------------------------nvec cossim 성능평가--------------------------#
        """ ------ nvec cossim 성능평가 ------
        

        #coods slicing
        way28_cood, way28_sigma, way28_coodBP = [], [], []
        for i in range(0, 500, 50):
            way28_cood.append(coods[28][i+25:i+35, :])
            way28_sigma.append(sigma[28][i+25:i+35])
            way28_coodBP.append(coods_BP[28][i+25:i+35, :])

        way28_cood = torch.cat(way28_cood, dim=0)
        way28_sigma = torch.cat(way28_sigma, dim=0)
        way28_coodBP = torch.cat(way28_coodBP, dim=0)
        occupied_idx = torch.where(way28_sigma > threshold)[0]

        csim = torch.nn.functional.cosine_similarity(global_num_grad[28], way28_coodBP[occupied_idx]).detach().numpy()
        csim_nvec = torch.nn.functional.cosine_similarity(torch.tensor([0, 0.02, 0]).to('cpu'), way28_coodBP[occupied_idx]).detach().numpy()

        #Compare Cos similarity
        bound = [-1.1, -0.8, 0.8, 1.1]
        count_between = np.count_nonzero( (np.digitize(csim, bound) == 1) | (np.digitize(csim, bound) == 3) )
        total_count = len(csim)
        print(f'기존 : {(count_between / total_count) * 100}%')
        cos_sim_occupied[28] = torch.nn.functional.cosine_similarity( torch.tensor([0, 0.02, 0]).to('cpu'), coods_BP[28][occupied_idx_sigma[28]], eps=1e-45).detach().numpy()
        count_between = np.count_nonzero( (np.digitize(csim_nvec, bound) == 1) | (np.digitize(csim_nvec, bound) == 3) )
        total_count = len(csim_nvec)
        print(f'변화 : {(count_between / total_count) * 100}%')

        pdb.set_trace()
        #coods slicing
        # coods[30][occupied_idx_sigma[30]] 보고 범위정하고 -> condition을 coods_BP에 적용
        condition2 = (coods[30][occupied_idx_sigma[30]][:, 0] >= -0.55) & (coods[30][occupied_idx_sigma[30]][:, 1] <= -0.60)
        coods[30][occupied_idx_sigma[30]][condition2]
        coods_BP[30][occupied_idx_sigma[30]][condition2]

        csim2 = torch.nn.functional.cosine_similarity(global_num_grad[30], coods_BP[30][occupied_idx_sigma[30]][condition2] ).detach().numpy()
        csim2_nvec = torch.nn.functional.cosine_similarity(torch.tensor([0, 0.02, 0]).to('cpu'), coods_BP[30][occupied_idx_sigma[30]][condition2]).detach().numpy()

        bound = [-1.1, -0.8, 0.8, 1.1]
        count_between = np.count_nonzero( (np.digitize(csim2, bound) == 1) | (np.digitize(csim2, bound) == 3) )
        total_count = len(csim)
        print(f'기존 : {(count_between / total_count) * 100}%')
        count_between = np.count_nonzero( (np.digitize(csim2_nvec, bound) == 1) | (np.digitize(csim2_nvec, bound) == 3) )
        total_count = len(csim_nvec)
        print(f'변화 : {(count_between / total_count) * 100}%')
        pdb.set_trace()
        """

        
        #--------------------------density csv--------------------------#
        """ ###### density csv
        
        np_density = density.cpu().detach().numpy()
        df = pd.DataFrame(data=np_density)
        df.to_csv("data_density.csv", index=False)

        df2 = pd.DataFrame(data=sigma_auto_array[12])
        df2.to_csv("data_bp_body.csv", index=False)

        df3 = pd.DataFrame(data=coods_array2[12])
        df3.to_csv("data_body.csv", index=False)

        df4 = pd.DataFrame(data=normalized_bp[12])
        df4.to_csv("data_bp_body_scaled.csv", index=False)

        colision_prob = torch.mean(density, dim = -1) 
        print(colision_prob)
        """
        
        #--------------------------Occupancy Graph--------------------------#
        """ Occupied region 표시하기
        
        """
        pdb.set_trace()
        print("Occupancy Graph")
        for way in range(28,29):#0,num_ways):
            tf_values = ( sigma[way] > threshold ).detach().numpy()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,12), subplot_kw={"projection":"3d"})

            for i in range(len(coods_array[way])):
                if tf_values[i]:
                    ax1.scatter(coods_array[way][i, 0], coods_array[way][i, 1], coods_array[way][i, 2], c='r', marker='o', alpha=0.5)
                else:
                    ax1.scatter(coods_array[way][i, 0], coods_array[way][i, 1], coods_array[way][i, 2], c='b', marker='x', alpha=0)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title(f"way{way}")
            # if way == 12: ax1.view_init(azim=120) #Stone Path1 180도 회전용 

            for i in range(len(coods_array[way])):
                if tf_values[i]:
                    ax2.scatter(coods_array[way][i, 0], coods_array[way][i, 1], coods_array[way][i, 2], c='r', marker='o', alpha=0.1)
                    ax2.quiver(coods_array[way][i, 0], coods_array[way][i, 1], coods_array[way][i, 2], 
                               coods_BP_normalize[way][i, 0], coods_BP_normalize[way][i, 1], coods_BP_normalize[way][i, 2], alpha=0.8)
                else:
                    ax2.scatter(coods_array[way][i, 0], coods_array[way][i, 1], coods_array[way][i, 2], c='b', marker='x', alpha=0)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f"way{way} vec")
            plt.show()
            # if way == 12: ax2.view_init(azim=120) #Stone Path1 180도 회전용 
            # if way == 30: plt.show()
            plt.savefig(f'plt/occupancy_way{way}.png')
            plt.close()

        #---------------------------histogram-----------------------------#
        
        pdb.set_trace()
        print("### hist save ###")
        for i in range(0,num_ways):
            data = cos_sim[i]
            bins = np.arange(-1, 1.1, 0.1)
            hist, bin_edges = np.histogram( data, bins )
            plt.figure(figsize=(10,6))
            plt.hist(data, bins=bins, edgecolor='k')
            plt.title(f'(Way{i}) Distribution of Cosine Similarity')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('distribution')
            plt.xticks(bin_edges)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(f'plt/body_histogram_way{i}.png')

            data = cos_sim_occupied[i]
            bins = np.arange(-1, 1.1, 0.1)
            hist, bin_edges = np.histogram( data, bins )
            plt.figure(figsize=(10,6))
            plt.hist(data, bins=bins, edgecolor='k')
            plt.title(f'(Way{i}) Distribution of Cosine Similarity')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('distribution')
            plt.xticks(bin_edges)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(f'plt/occupied_histogram_way{i}.png')
            plt.close()
        
        ################### ani ##################
        pdb.set_trace()
        print("### gif save ###")

        fig, axs = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={"projection":"3d"})
        fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
        def init():
            for ax in axs:
                zdata = f"way 14 cossim" if ax == axs[0] else "way 14 occupied"
                ax.set_xlabel("X", fontdict=fontlabel, labelpad=14)
                ax.set_ylabel("Y", fontdict=fontlabel, labelpad=14)
                ax.set_title(zdata, fontdict=fontlabel)
                
                ### body path
            # axs[0].quiver(coods_array2[10, :, 0], coods_array2[10, :, 1], coods_array2[10, :, 2],
            #             bp_norm_scaled[10, :, 0], bp_norm_scaled[10, :, 1], bp_norm_scaled[10, :, 2])
            # axs[1].quiver(coods_array2[14, :, 0], coods_array2[14, :, 1], coods_array2[14, :, 2],
            #             bp_norm_scaled[14, :, 0], bp_norm_scaled[14, :, 1], bp_norm_scaled[14, :, 2])
            
            axs[0].quiver(coods_array2[14][indices_csim_way14 == 3][:, 0], coods_array2[14][indices_csim_way14 == 3][:, 1], coods_array2[14][indices_csim_way14 == 3][:, 2],
                          bp_norm_scaled[14][indices_csim_way14 == 3][:, 0], bp_norm_scaled[14][indices_csim_way14 == 3][:, 1], bp_norm_scaled[14][indices_csim_way14 == 3][:, 2])
            # Warning : way index 변경 시 indices_csim 계산의 index도 같이 변경해줘야함
            axs[1].quiver(coods_array2[14][indices_sigma_14][:, 0], coods_array2[14][indices_sigma_14][:, 1], coods_array2[14][indices_sigma_14][:, 2],
                          bp_norm_scaled[14][indices_sigma_14][:, 0], bp_norm_scaled[14][indices_sigma_14][:, 1], bp_norm_scaled[14][indices_sigma_14][:, 2])
            # axs[1].quiver(0.0228, 0.3001, 0.1986,
            #               -0.0008, 0.0688, 0.0005)
            # axs[1].quiver(-0.0162, 0.2945, 0.2201,
            #               0.0114, 0.0889, -0.0398)
            return fig,
        def animate(i):
            axs[0].view_init(elev=30., azim=i)
            axs[1].view_init(elev=30., azim=i)
            return fig,
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=False)
        
        anim.save('3d_scatter_way14.gif', fps=30)

        

        
        pdb.set_trace()
        ################## 10.25 ##########################


        """ Min Max의 평균vec에 대한 SIM 0.7이상에 대해서 sigma 
        """
        cst0_similarity = torch.nn.functional.cosine_similarity(numerical_grad_vec, max_grad, eps=1e-45)
        cst0_similarity = cst0_similarity.cpu().detach().numpy() 

        bound = [-0.2, 0.2, 0.7, 1.0]
        indices = np.digitize(cst0_similarity, bound)
        top_den = top_den.cpu().detach().numpy()

        for i in range(1, 4):
            print(cst0_similarity[indices == i].shape)
            plt.figure(figsize=(8,6))
            plt.scatter(cst0_similarity[indices == i], top_den[indices == i], alpha=0.5)
            plt.xlabel('similarity')
            plt.ylabel('sigma, logscale')
            plt.title('similarity sigma plot')
            plt.grid(True)
            
            # 숫자 1을 변수 i로 바꾸어 파일명 생성
            filename = f'plt/mean_sigma{i}.png'
            # 저장
            plt.savefig(filename)

        """ Min Max vec들에 대한 sim 0.7이상에 대해서 sigma -0.2~0.2, 특성 특정범위에 몰려있다던가 하는 분석
        """ 
        cst1_similarity = torch.nn.functional.cosine_similarity(numerical_grad_vectors, max_grad, eps=1e-45)
        cst1_similarity = cst1_similarity.cpu().detach().numpy()

        bound = [-0.2, 0.2, 0.7, 1.0]
        indices = np.digitize(cst1_similarity, bound)
        # top_den = top_den.cpu().detach().numpy() # 중복방지

        for i in range(1,4):
            plt.figure(figsize=(8,6))
            plt.scatter(cst1_similarity[indices == i], top_den[indices == i], alpha=0.5)
            plt.xlabel('similarity')
            plt.ylabel('sigma, logscale')
            plt.title('similarity sigma plot')
            plt.grid(True)
            
            # 숫자 1을 변수 i로 바꾸어 파일명 생성
            filename = f'plt/nomal_sigma{i}.png'
            # 저장
            plt.savefig(filename)

        """ 22개의 sim에 대한 index에 대해 density 뽑기
        for i in range(1,22):
            plt.figure(figsize=(8,6))
            plt.scatter(similarity[indices == i], density_flat_log[indices == i], alpha=0.5)
            plt.xlabel('similarity')
            plt.ylabel('sigma, logscale')
            plt.title('similarity sigma plot')
            plt.grid(True)
            
            # plt.savefig('bp_x_sim_sigma_plot.png')
            # plt.savefig('plt/x_sim_sigma1.png')
            # plt.show()
            
            # 숫자 1을 변수 i로 바꾸어 파일명 생성
            filename = f'plt/x_sim_sigma{i}.png'
            # 저장
            plt.savefig(filename)

        data = similarity
        bins = np.arange(-1, 1.1, 0.1)
        hist, bin_edges = np.histogram(data, bins)
        plt.figure(figsize=(10,6))
        plt.hist(data, bins=bins, edgecolor='k')
        plt.title('Distribution of Cosine Similarity')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('distribution')
        plt.xticks(bin_edges)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('bp_x_cos_sim.png')
        # plt.show()
        """
        

        """ 원본 코드
        valid_ind = np.isfinite(similarity) & np.isfinite(density_flat_log)
        sim = similarity[valid_ind]
        sigma = density_flat_log[valid_ind]

        plt.figure(figsize=(8,6))
        plt.scatter(sim, sigma, alpha=0.5)
        plt.xlabel('similarity')
        plt.ylabel('sigma, logscale')
        plt.title('similarity sigma plot')
        plt.grid(True)

        plt.savefig('bp_z_sim_sigma_plot.png')
        

        data = similarity
        bins = np.arange(-1, 1.1, 0.1)
        hist, bin_edges = np.histogram(data, bins)
        plt.figure(figsize=(10,6))
        plt.hist(data, bins=bins, edgecolor='k')
        plt.title('Distribution of Cosine Similarity')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('distribution')
        plt.xticks(bin_edges)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('bp_z_cos_sim.png')
        """

        print(" ### state cost funct end ### ")
        pdb.set_trace()
        
        # multiplied by distance to prevent it from just speed tunnelling
        # S =   S,B * S,_
        colision_prob = torch.mean(density * distance[:,None], dim = -1) 

        if self.epoch < self.fade_out_epoch:
            t = torch.linspace(0,1, colision_prob.shape[0])
            position = self.epoch/self.fade_out_epoch
            mask = torch.sigmoid(self.fade_out_sharpness * (position - t)).to(device)
            colision_prob = colision_prob * mask

        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**4 + colision_prob * 1e6, colision_prob*1e6

    def total_cost(self):
        total_cost, colision_loss  = self.get_state_cost()
        return torch.mean(total_cost)

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True) 
        #opt = torch.optim.Adam(self.params(), lr=self.lr)
        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it
                loss = self.total_cost()
                #pdb.set_trace()
                self.init_loss_log.append(loss.item())
                print(str(it) + " iteration (" +  str(loss.item()) + ")")
                loss.backward()
                opt.step()

                save_step = 50
                if it%save_step == 0:
                    if hasattr(self, "basefolder"):
                        self.save_poses(self.basefolder / "init_poses" / (str(it//save_step)+".json"))
                        self.save_costs(self.basefolder / "init_costs" / (str(it//save_step)+".json"))
                    else:
                        print("Warning: data not saved!")

            #### Save Init Loss Log as csv ####
            filename = './loss_log/init_loss_log.csv'
            with open(filename, 'w', newline='') as f : 
                writer = csv.writer(f) 
                writer.writerow(self.init_loss_log) 
            
            pdb.set_trace()
            loss_data = np.array(self.init_loss_log)
            loss_data_shift = np.concatenate((loss_data[1:], np.array([loss_data[-1]])))
            loss_grad = np.abs(loss_data - loss_data_shift)
                
            x = np.linspace(0,2499, 2500) 
            y = loss_grad
            plt.plot(x, y)
            plt.savefig('loss_grad.png')
            
            pdb.set_trace()
            iter_x = np.linspace(0, self.epochs_init-1, self.epochs_init)
            err_y = np.array(self.init_loss_log)
            df = pd.DataFrame(iter_x, columns=['iteration'])
            df['loss'] = err_y
            #df.to_csv("loss_iter_table.csv", index=False)
            

        except KeyboardInterrupt:
            print("finishing early")

    def learn_update(self, iteration):
        print(iteration," th Learn updating: ", self.epochs_update)
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)

        for it in range(self.epochs_update):
            opt.zero_grad()
            self.epoch = it
            loss = self.total_cost()
            print(it, loss)
            loss.backward()
            opt.step()
            # it += 1

            # if (it > self.epochs_update and self.max_residual < 1e-3):
            #     break

            save_step = 50
            if it%save_step == 0:
                if hasattr(self, "basefolder"):
                    self.save_poses(self.basefolder / "replan_poses" / (str(it//save_step)+ f"_time{iteration}.json"))
                    self.save_costs(self.basefolder / "replan_costs" / (str(it//save_step)+ f"_time{iteration}.json"))
                else:
                    print("Warning: data not saved!")

        print("Learned matrix: ", self.states)
        print("Learned shape: ", self.states.shape)

    def update_state(self, measured_state):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        self.start_state = measured_state
        self.states = self.states[1:, :].detach().requires_grad_(True)
        self.initial_accel = actions[1:3, 0].detach().requires_grad_(True)
        print("New state : ", self.states)
        print("New state shape: ", self.states.shape)
        # print(self.initial_accel.shape)

    def plot(self, quadplot):
        quadplot.trajectory( self, "g" )
        ax = quadplot.ax_graph

        pos, vel, accel, _, omega, _, actions = self.calc_everything()
        actions = actions.cpu().detach().numpy()
        pos = pos.cpu().detach().numpy()
        vel = vel.cpu().detach().numpy()
        omega = omega.cpu().detach().numpy()

        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        ax.plot(pos[...,0], label="px")
        # ax.plot(pos[...,1], label="py")
        # ax.plot(pos[...,2], label="pz")

        ax.plot(vel[...,0], label="vx")
        # ax.plot(vel[...,1], label="vy")
        ax.plot(vel[...,2], label="vz")

        # ax.plot(omega[...,0], label="omx")
        ax.plot(omega[...,1], label="omy")
        # ax.plot(omega[...,2], label="omz")

        ax_right = quadplot.ax_graph_right

        total_cost, colision_loss = self.get_state_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()

    def save_poses(self, filename):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        poses = []
        pose_dict = {}
        with open(filename,"w+") as f:
            for pos, rot in zip(positions, rot_matrix):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.cpu().detach().numpy()
                pose[:3, 3]  = pos.cpu().detach().numpy()
                pose[3,3] = 1

                poses.append(pose.tolist())
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

    def save_costs(self, filename):
        positions, vel, _, rot_matrix, omega, _, actions = self.calc_everything()
        total_cost, colision_loss  = self.get_state_cost()

        output = {"colision_loss": colision_loss.cpu().detach().numpy().tolist(),
                  "pos": positions.cpu().detach().numpy().tolist(),
                  "actions": actions.cpu().detach().numpy().tolist(),
                  "total_cost": total_cost.cpu().detach().numpy().tolist()}

        with open(filename,"w+") as f:
            json.dump( output,  f, indent=4)

    def save_progress(self, filename):
        if hasattr(self.renderer, "config_filename"):
            config_filename = self.renderer.config_filename
        else:
            config_filename = None

        to_save = {"cfg": self.cfg,
                    "start_state": self.start_state,
                    "end_state": self.end_state,
                    "states": self.states,
                    "initial_accel":self.initial_accel,
                    "config_filename": config_filename,
                    }
        torch.save(to_save, filename)


    # def load_progress(cls, filename, renderer=None):
    #     # a note about loading: it won't load the optimiser learned step sizes
    #     # so the first couple gradient steps can be quite bad

    #     loaded_dict = torch.load(filename)
    #     print(loaded_dict)

    #     if renderer == None:
    #         assert loaded_dict['config_filename'] is not None
    #         renderer = load_nerf(loaded_dict['config_filename'])

    #     obj = cls(renderer, loaded_dict['start_state'], loaded_dict['end_state'], loaded_dict['cfg'])
    #     obj.states = loaded_dict['states'].requires_grad_(True)
    #     obj.initial_accel = loaded_dict['initial_accel'].requires_grad_(True)

    #     return obj

'''
def main():

    # violin - astar
    # renderer = get_nerf('configs/violin.txt')
    # start_state = torch.tensor([0.44, -0.23, 0.2, 0])
    # end_state = torch.tensor([-0.58, 0.66, 0.15, 0])

    #playground
    experiment_name = "playground_slide"
    renderer = get_nerf('configs/playground.txt')

    # under slide
    start_pos = torch.tensor([-0.3, -0.27, 0.06])
    end_pos = torch.tensor([0.02, 0.58, 0.65])

    # around slide
    # start_pos = torch.tensor([-0.3, -0.27, 0.06])
    # end_pos = torch.tensor([-0.14, 0.6, 0.78])


    #stonehenge
    # renderer = get_nerf('configs/stonehenge.txt')
    # start_state = torch.tensor([-0.06, -0.79, 0.2, 0])
    # end_state = torch.tensor([-0.46, 0.55, 0.16, 0])

    # start_pos = torch.tensor([-0.05,-0.9, 0.2])
    # end_pos   = torch.tensor([-1 , 0.7, 0.35])
    # start_pos = torch.tensor([-1, 0, 0.2])
    # end_pos   = torch.tensor([ 1, 0, 0.5])


    start_R = vec_to_rot_matrix( torch.tensor([0.0,0.0,0]))
    start_state = torch.cat( [start_pos, torch.tensor([0,0,0]), start_R.reshape(-1), torch.zeros(3)], dim=0 )
    end_state   = torch.cat( [end_pos,   torch.zeros(3), torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )

    # experiment_name = "test" 
    # filename = "line.plan"
    # renderer = get_manual_nerf("empty")
    # renderer = get_manual_nerf("cylinder")

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.01,
            "epochs_init": 2500,
            "fade_out_epoch": 0,
            "fade_out_sharpness": 10,
            "epochs_update": 250,
            }


    basefolder = "experiments" / pathlib.Path(experiment_name)
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "train_poses").mkdir()
    (basefolder / "train_graph").mkdir()

    print("created", basefolder)

    traj = System(renderer, start_state, end_state, cfg)
    # traj = System.load_progress(filename, renderer); traj.epochs_update = 250 #change depending on noise

    traj.basefolder = basefolder

    traj.a_star_init()

    # quadplot = QuadPlot()
    # traj.plot(quadplot)
    # quadplot.show()

    traj.learn_init()

    traj.save_progress(basefolder / "trajectory.pt")

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.show()


    save = Simulator(start_state)
    save.copy_states(traj.get_full_states())

    if False: # for mpc control
        sim = Simulator(start_state)
        sim.dt = traj.dt #Sim time step changes best on number of steps

        for step in range(cfg['steps']):
            action = traj.get_next_action().clone().detach()
            print(action)

            state_noise = torch.normal(mean= 0, std=torch.tensor( [0.01]*3 + [0.01]*3 + [0]*9 + [0.005]*3 ))
            # state_noise[3] += 0.0 #crosswind

            # sim.advance(action) # no noise
            sim.advance(action, state_noise) #add noise
            measured_state = sim.get_current_state().clone().detach()

            measurement_noise = torch.normal(mean= 0, std=torch.tensor( [0.01]*3 + [0.02]*3 + [0]*9 + [0.005]*3 ))
            measured_state += measurement_noise
            traj.update_state(measured_state) 

            traj.learn_update()

            print("sim step", step)
            if step % 5 !=0 or step == 0:
                continue

            quadplot = QuadPlot()
            traj.plot(quadplot)
            quadplot.trajectory( sim, "r" )
            quadplot.trajectory( save, "b", show_cloud=False )
            quadplot.show()


        quadplot = QuadPlot()
        traj.plot(quadplot)
        quadplot.trajectory( sim, "r" )
        quadplot.trajectory( save, "b", show_cloud=False )
        quadplot.show()

def OPEN_LOOP(traj):
    sim = Simulator(traj.start_state)
    sim.dt = traj.dt #Sim time step changes best on number of steps

    for step in range(cfg['steps']):
        action = traj.get_actions()[step,:].detach()
        print(action)
        sim.advance(action)

    quadplot = QuadPlot()
    traj.plot(quadplot)
    quadplot.trajectory( sim, "r" )
    quadplot.trajectory( save, "b", show_cloud=False )
    quadplot.show()

if __name__ == "__main__":
    main()
'''
