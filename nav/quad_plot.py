import torch
from torch._C import device
import numpy as np
import json

from .math_utils import rot_matrix_to_vec
from .quad_helpers import *#astar, next_rotation, 2d_astar
import pdb
import matplotlib.pyplot as plt

import time
import cv2

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Planner:
    def __init__(self, start_state, end_state, cfg, density_fn, opt):
        self.nerf = density_fn
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
        
        ### Coarse A* & BP & Num Grad Way Points Coordinate ###
        self.states = states.clone().detach().requires_grad_(True)

        ### BP ONLY Way Points Coordinate ###
        self.bp_states = states.clone().detach().requires_grad_(True)

        self.occupancy = None
        
        self.initial_accel = torch.tensor([cfg['g'], cfg['g']]).requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(self.body_extent[0, 0], self.body_extent[0, 1], self.body_nbins[0]),
                                            torch.linspace(self.body_extent[1, 0], self.body_extent[1, 1], self.body_nbins[1]),
                                            torch.linspace(self.body_extent[2, 0], self.body_extent[2, 1], self.body_nbins[2])), dim=-1)
        self.robot_body = body.reshape(-1, 3)

        if self.CHURCH:
            self.robot_body = self.robot_body/2

        self.epoch = 0

        self.hybrid_dist = 0
        self.bp_dist = 0

        self.hybrid_iters = 0
        self.bp_iters = 0

    def full_to_reduced_state(self, state):
        pos = state[:3]
        R = state[6:15].reshape((3,3))

        x,y,_ = R @ torch.tensor( [1.0, 0, 0 ] )
        angle = torch.atan2(y, x)

        return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()

    def params(self):
        return [self.initial_accel, self.states]

    def calc_everything(self):

        start_pos   = self.start_state[None, 0:3]
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
        current_pos = torch.cat( [start_pos, next_pos, after_next_pos, after2_next_pos, self.states[2:, :3], end_pos], dim=0)

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

        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################
        
        # pos_dh 쓰는이유 : pos, cost json save호출 시 path를 온전히 반영하려고
        current_pos_dh = torch.cat( [start_pos, start_pos, self.states[:, :3], end_pos], dim=0)
        
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

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_state_cost(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0].to(device)
        torques = torch.norm(actions[:, 1:], dim=-1).to(device)

        # S, B, 3  =  S, _, 3 +      _, B, 3   X    S, _,  3
        B_body, B_omega = torch.broadcast_tensors(self.robot_body, omega[:,None,:])
        point_vels = vel[:,None,:] + torch.cross(B_body, B_omega, dim=-1)

        # S, B
        distance = torch.sum( vel**2 + 1e-5, dim = -1)**0.5
        # S, B
        density = self.nerf( self.body_to_world(self.robot_body) )**2

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

        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it
                loss = self.total_cost()
                print(it, loss)
                loss.backward()

                opt.step()

                save_step = 50
                if it%save_step == 0:
                    if hasattr(self, "basefolder"):
                        self.save_poses(self.basefolder / "init_poses" / (str(it//save_step)+".json"))
                        self.save_costs(self.basefolder / "init_costs" / (str(it//save_step)+".json"))
                    else:
                        print("Warning: data not saved!")
                ####################### ##########################################
                pdb.set_trace()
                ####################### ##########################################
            

        except KeyboardInterrupt:
            print("finishing early")
        ####################### ##########################################
        pdb.set_trace()
        ####################### ##########################################
        
    def learn_update(self, iteration):
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

    def update_state(self, measured_state):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        self.start_state = measured_state
        self.states = self.states[1:, :].detach().requires_grad_(True)
        self.initial_accel = actions[1:3, 0].detach().requires_grad_(True)
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

    def save_poses(self, filename, mode=None):
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
                # print(f'pos {pos}')
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

            total_L2_distance = 0.0
            for i in range(0, positions.shape[0] - 1):
                L2_distance = torch.dist(positions[i], positions[i+1], p=2)
                total_L2_distance += L2_distance
            print(f'total distance = {total_L2_distance:.4}')

        if mode == "hybrid":
            self.hybrid_dist = total_L2_distance
        else:
            self.bp_dist = total_L2_distance

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


######################################## DH Custom funct #################################################
    
    def a_star_init(self):
        side = self.opt.a_star_grid *5 #100 #PARAM grid size
        
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
        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

        # 20, 20, 20
        occupied = maxpool(output[None,None,...])[0,0,...] > 0.3

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        print(start, end)
        #path = astar(occupied, start, end)
        path = astar(occupied, start, end, self.nerf, self.robot_body, side)

        # convert from index cooredinates
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        #adding way
        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

        #prevents weird zero derivative issues
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        states = (prev_smooth + next_smooth + states)/3

        self.states = states.clone().detach().requires_grad_(True)

        self.save_poses(self.basefolder / "init_poses" / (str(f"A_star_path")+".json"))

        self.hybrid_sampling_states = states.clone().detach() 
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        self.hybrid_posi = positions.clone().detach()
        #nerf_loss = self.dh_get_cost()
        #print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')
    def interp_init(self, start, end, points):
        side = 100 #PARAM grid size
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
        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

        # 20, 20, 20
        occupied = maxpool(output[None,None,...])[0,0,...] > 0.3


        self.start_pos = start.cpu().numpy()
        self.end_pos = end.cpu().numpy()


        num_points = points
        delta = self.end_pos - self.start_pos
        intermediate_points = [self.start_pos + i * delta / (num_points + 1) for i in range(1, num_points + 1)]

        intermediate_points_tensor = np.array(intermediate_points, dtype=np.float32)
        intermediate_points_tensor = torch.tensor(intermediate_points_tensor, dtype=torch.float32)
        
        states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1)
        
        
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        self.states = states.clone().detach().requires_grad_(True)
        self.save_poses(self.basefolder / "init_poses" / (str(f"A_star_interp")+".json"))
    def hybrid_init(self):
        side = self.opt.a_star_grid *5 #100 #PARAM grid size
        
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
        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

        # 20, 20, 20
        occupied = maxpool(output[None,None,...])[0,0,...] > 0.3

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        print(start, end)
        
        path = astar(occupied, start, end)
        
        # convert from index cooredinates
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        #adding way
        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

        #prevents weird zero derivative issues
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness
        self.states = states.clone().detach().requires_grad_(True)
        
        ######
        nerf_loss = self.dh_get_cost()
        print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')
        
        total_ways = nerf_loss > 0.3
        true_indices = torch.where(total_ways)[0]
        print(f'occ idx : {true_indices}')
        
        
        tensor = states[:, :3]
        
        result_tensor = []
        for i in range(0, len(tensor) - 1, 1):
            if torch.any(i == true_indices):
                if i + 2 < len(tensor):
                    result_tensor.append(self.interp_tensor(tensor[i:i+2], 5))
                elif i == len(tensor) - 2:
                    result_tensor.append(self.interp_tensor(tensor[-2:], 5))
                elif i == len(tensor) - 1:
                    result_tensor.append(tensor[-1])
            else:
                if i + 2 < len(tensor):
                    result_tensor.append(self.interp_tensor(tensor[i:i+2], 0))
                elif i == len(tensor) - 2:
                    result_tensor.append(self.interp_tensor(tensor[-2:], 0))
                elif i == len(tensor) - 1:
                    result_tensor.append(tensor[-1])
            
        result_tensor = torch.cat(result_tensor, dim=0)
        
        states_interp = torch.cat( [result_tensor, torch.zeros( (result_tensor.shape[0], 1) ) ], dim=-1)
        self.states = states_interp.clone().detach().requires_grad_(True)
        self.save_poses(self.basefolder / "init_poses" / (str(f"A_star_hybrid")+".json"))
    def interp_dense_init(self, start, end, points):
        self.start_pos = start.cpu().numpy()
        self.end_pos = end.cpu().numpy()


        num_points = points
        delta = self.end_pos - self.start_pos
        intermediate_points = [self.start_pos + i * delta / (num_points + 1) for i in range(1, num_points + 1)]

        intermediate_points_tensor = np.array(intermediate_points, dtype=np.float32)
        intermediate_points_tensor = torch.tensor(intermediate_points_tensor, dtype=torch.float32)
        
        states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1)
        
        
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness
        self.states = states.clone().detach().requires_grad_(True)
        
        
        
        nerf_loss = self.dh_get_cost()
        print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(states)}')
        
        total_ways = nerf_loss > 0.3
        true_indices = torch.where(total_ways)[0]
        print(f'occ idx : {true_indices}')
        
        tensor = states[:, :3]
        
        result_tensor = []
        for i in range(0, len(tensor) - 1, 1):
            if torch.any(i == true_indices):
                if i + 2 < len(tensor):
                    result_tensor.append(self.interp_tensor(tensor[i:i+2], 5))
                elif i == len(tensor) - 2:
                    result_tensor.append(self.interp_tensor(tensor[-2:], 5))
                elif i == len(tensor) - 1:
                    result_tensor.append(tensor[-1])
            else:
                if i + 2 < len(tensor):
                    result_tensor.append(self.interp_tensor(tensor[i:i+2], 0))
                elif i == len(tensor) - 2:
                    result_tensor.append(self.interp_tensor(tensor[-2:], 0))
                elif i == len(tensor) - 1:
                    result_tensor.append(tensor[-1])
            
        result_tensor = torch.cat(result_tensor, dim=0)
        
        states_interp = torch.cat( [result_tensor, torch.zeros( (result_tensor.shape[0], 1) ) ], dim=-1)
        self.states = states_interp.clone().detach().requires_grad_(True)
        self.save_poses(self.basefolder / "init_poses" / (str(f"A_star_interp")+".json"))
        
    
    def interp_tensor(self, tensor, num_points):
        def linear_interpolation(point1, point2, steps):
            return [point1 + i * (point2 - point1) / (steps - 1) for i in range(steps)]

        result_tensor = []
        for i in range(len(tensor) - 1):
            result_tensor.append(tensor[i])
            result_tensor.extend(linear_interpolation(tensor[i], tensor[i + 1], num_points + 2)[1:-1])
        result_tensor.append(tensor[-1])
        result_tensor = torch.stack(result_tensor)
        return result_tensor

    def a_star_sdf(self):

        side = self.opt.a_star_grid*5 #PARAM grid size 

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
        grid_size = side//kernel_size

        #convert to index cooredinates
        self.start_state[:3] = torch.tensor( self.opt.start_pos )
        self.end_state  [:3] = torch.tensor(self.opt.end_pos)
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )



        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)
        
        # 20, 20, 20
        occupied_value = maxpool(output[None,None,...])[0,0,...]
        occupied = occupied_value > 0.3

        pdb.set_trace()
        import pandas as pd
        data = pd.read_csv( '/home/sslunder0/project/sc/work/nerf-navi/scannet/mesh_stage0/Sigmas_Reso100_quant.csv', header=None  )
        data = torch.tensor( data.values.reshape(100, 100, 100) )
        sdf_threshold = 0.01
        sdf_occupied_value = self.dh_MinPool3d(in_tensor=data)
        sdf_occupied = sdf_occupied_value < sdf_threshold

        pd.DataFrame(data=occupied.reshape(-1, occupied.shape[1]).cpu().detach().numpy()).to_csv('/home/sslunder0/project/nerf-navigation-num-grad/paths/st_scannetroom/init_poses/sigmas.csv', index=False)
        pd.DataFrame(data=sdf_occupied.reshape(-1, sdf_occupied.shape[1]).cpu().detach().numpy()).to_csv('/home/sslunder0/project/nerf-navigation-num-grad/paths/st_scannetroom/init_poses/sdf_sigmas.csv', index=False)
        

        pdb.set_trace()
        self.dh_Encoding_mem_req(bitmap_tensor=sdf_occupied, bitmap_threshold=sdf_threshold)




        pdb.set_trace()
        z_bound = 15
        path = astar(sdf_occupied, start, end, self.nerf, self.robot_body, side, z_bound, self.opt.path[5:len(self.opt.path)-1], self.opt)
        ##---------- 3) path save -------------##
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1
        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

        self.start_state[None, 0:3] = states[0, :3]
        self.end_state[None, 0:3] = states[-1, :3]

        #path info
        print('\033[31m \033[43m'+f'path'+ '\033[0m' + f', A* IDX:{start, end}, Start --> End Pos:{states[0, :3], states[-1, :3]}, waypoints:{len(path)}')
        print('\033[31m \033[43m'+f'path'+ '\033[0m' + f', A* path: {path}')

        ##---------- 4) smooth path (diagram of which states are averaged) -------------## -> A* 에선 필요없음
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        #prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        #next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        #states = (prev_smooth + next_smooth + states)/3

        ### Set A Star Wayp Coordinate ###
        self.states = states.clone().detach().requires_grad_(True)
        self.bp_states = states.clone().detach().requires_grad_(True)

        self.save_poses(self.basefolder / "init_poses" / (str(f"a_star_path")+".json"))
        
        # A* path colision info
        nerf_loss = self.dh_get_cost()
        print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')
        
        return

    def dh_a_star_init_random(self):
        side = self.opt.a_star_grid*5 #PARAM grid size 

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
        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)
        
        # 20, 20, 20
        occupied_value = maxpool(output[None,None,...])[0,0,...]
        occupied = occupied_value > 0.3

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )


        if self.opt.random_path:
            wayp_random_flag = True
            wayp_random_count = 0
            wayp_failed_count = 0
            wayp_collision_count = 0
            total_path_dist = 0
            while wayp_random_flag:

                z_bound = self.sr_set_zbound(path = self.opt.path[5:len(self.opt.path)])

                if self.opt.path[5:len(self.opt.path)] == "stonehenge":
                    z_bound = 14
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                if self.opt.path[5:len(self.opt.path)] == "scannetroom":
                    z_bound = 16
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                if self.opt.path[5:len(self.opt.path)] == "scannet_472":
                    z_bound = 14
                    start_pos = torch.tensor([np.random.uniform(7, 14), np.random.uniform(1, 19), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(7, 14), np.random.uniform(1, 19), np.random.uniform(10, z_bound)])
                if self.opt.path[5:len(self.opt.path)] == "scala_room":
                    z_bound = 14
                    start_pos = torch.tensor([np.random.uniform(2, 18), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(2, 18), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                if self.opt.path[5:len(self.opt.path)] == "replica_FRL_v2":
                    z_bound = 15
                    start_pos = torch.tensor([np.random.uniform(5, 15), np.random.uniform(1, 20), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(5, 15), np.random.uniform(1, 20), np.random.uniform(10, z_bound)])
                if self.opt.path[5:len(self.opt.path)] == "replica_office3":
                    z_bound = 16
                    start_pos = torch.tensor([np.random.uniform(1, 19), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(1, 19), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                if self.opt.path[5:len(self.opt.path)] == "replica_apt2":
                    z_bound = 15
                    start_pos = torch.tensor([np.random.uniform(3, 19), np.random.uniform(0, 19), np.random.uniform(10, z_bound)])  #* side / 100
                    end_pos = torch.tensor([np.random.uniform(3, 19), np.random.uniform(0, 19), np.random.uniform(10, z_bound)])  #* side / 100
                if self.opt.path[5:len(self.opt.path)] == "replica_apt1":
                    z_bound = 15
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(3, 16), np.random.uniform(10, z_bound)])  #* side / 100
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(3, 16), np.random.uniform(10, z_bound)])  #* side / 100
                if self.opt.path[5:len(self.opt.path)] == "replica_hotel":
                    z_bound = 16
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(7, 15), np.random.uniform(10, z_bound)])  #* side / 100
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(7, 15), np.random.uniform(10, z_bound)])  #* side / 100
                

                self.start_pos = np.array(start_pos.cpu())
                self.end_pos = np.array(end_pos.cpu())

                if side == 200:
                    # Start/End Point Matching on 20 -> 40 Grid
                    start = tuple(int(start_pos[i])  for i in range(3) ) 
                    end = tuple(int(end_pos[i])  for i in range(3) ) 
                    start = tuple(x * 2 for x in start)  
                    end = tuple(x * 2 for x in end)  
                else:
                    start = tuple(int(start_pos[i] * side / 100 )  for i in range(3) ) 
                    end = tuple(int(end_pos[i] * side / 100)  for i in range(3) ) 


                ### 정사각형 Map 내에서 특정 영역의 start/end를 제외
                if self.opt.path[5:len(self.opt.path)] == "replica_FRL_v2":
                    x_max = 10 * side / 100
                    y_max = 7 * side / 100  
                    if (start[0] < x_max and start[1] < y_max) or (end[0] < x_max and end[1] < y_max): #FRL apt0
                        continue
                # if self.opt.path[5:len(self.opt.path)] == "replica_apt2": #주석이유 : 방내에만 한정하려면 for문 800횟수이상씩 돌아야해서
                #     x_max = 12 * side / 100
                #     y_min = 3 * side / 100  
                #     if (start[0] < x_max and start[1] > y_min) or (end[0] < x_max and end[1] > y_min): #rep apt2
                #         print(f"path{idx} occupied")
                #         continue
                if self.opt.path[5:len(self.opt.path)] == "scannet_472":
                    x_max = 8 * side / 100
                    y_max = 9 * side / 100  
                    if (start[0] < x_max and start[1] < y_max) or (end[0] < x_max and end[1] < y_max): #scannet 472
                        continue




                if occupied[start]:
                    #print(f"path start {start} waypoint occupied !!!")
                    continue
                elif occupied[end]:
                    #print(f"path end {end} waypoint occupied !!!")
                    continue
                elif start == end:
                    continue
                path = astar_extend(occupied, start, end, self.nerf, self.robot_body, side, z_bound, self.opt.path[5:len(self.opt.path)-1], self.opt)
                
                if path == None:
                    #print(f'\033[31m path S/E colision! \033[0m')
                    continue
                elif path == 0:
                    print(f'\033[31m path{wayp_random_count} Failed! \033[0m')

                    #loop condition check
                    wayp_random_count += 1
                    wayp_failed_count += 1
                    wayp_random_flag = (wayp_random_count != self.opt.total_path_num)
                    if not wayp_random_flag:
                        print('\033[31m'+f'##--- Total Path Distance : {total_path_dist:.3}, Mean Path Distance : {total_path_dist/(wayp_random_count - wayp_failed_count):.3}---##'+'\033[0m')
                        print(f'Success Paths : {self.opt.total_path_num - ( wayp_failed_count + wayp_collision_count )} ')
                    continue


                squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1
                states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

                self.start_state[None, 0:3] = states[0, :3]
                self.end_state[None, 0:3] = states[-1, :3]

                self.states = states.clone().detach().requires_grad_(True)
                self.bp_states = states.clone().detach().requires_grad_(True)

                #self.save_poses(self.basefolder / "init_poses" / (str(f"a_star_path{wayp_random_count}")+".json"))
                path_dist = self.dh_save_poses_L2_dist(self.basefolder / "init_poses" / (str(f"a_star_path")+".json"))
                total_path_dist += path_dist

                #path info
                print(f'Path {wayp_random_count} Start!')

                print('\033[31m \033[43m'+f'path{wayp_random_count}'+ '\033[0m' + f', A* IDX:{start, end}, Start --> End Pos:{states[0, :3], states[-1, :3]}, waypoints:{len(path)}')
                #print('\033[31m \033[43m'+f'path{wayp_random_count}'+ '\033[0m' + f', A* path: {path}')
                # nerf_loss = self.dh_get_cost()
                # print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')
                
                if self.dh_collision_info(tensor = states[:, :3], num_inp = 11):
                    wayp_collision_count += 1
                    print('\033[31m'+f'Colision!!!'+'\033[0m')
                else:
                    print(f'Not!')
                #loop condition check
                wayp_random_count += 1
                wayp_random_flag = (wayp_random_count != self.opt.total_path_num)

                if not wayp_random_flag:
                    print('\033[31m'+f'##--- Total Path Distance : {total_path_dist:.3}, Mean Path Distance : {total_path_dist/(wayp_random_count - wayp_failed_count):.3}---##'+'\033[0m')
                    print(f'Success Paths : {self.opt.total_path_num - ( wayp_failed_count + wayp_collision_count )} ')
                    

        elif not self.opt.random_path:
            # Set Start/End Points Manually
            print(f'Set Start/End Points by Arguments')    
            start_pos = torch.tensor( [self.opt.start_pos[0] , self.opt.start_pos[1] , self.opt.start_pos[2]] )
            end_pos = torch.tensor( [self.opt.end_pos[0] , self.opt.end_pos[1] , self.opt.end_pos[2]] )
            #convert to index cooredinates
            start_grid_float = grid_size*(start_pos[:3] + 1)/2       # Convert coord to 0~grid_size
            end_grid_float   = grid_size*(end_pos  [:3] + 1)/2
            start = tuple(int(start_grid_float[i]) for i in range(3) )
            end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

            self.start_pos = np.array(start_pos.cpu())
            self.end_pos = np.array(end_pos.cpu())


            z_bound = self.sr_set_zbound(path = self.opt.path[5:len(self.opt.path)])

            path = astar(occupied, start, end, self.nerf, self.robot_body, side, z_bound, self.opt.path[5:len(self.opt.path)-1], self.opt)
                
            if path == None:
                print(f'\033[31m path S/E colision! \033[0m')
                return
            elif path == 0:
                print(f'\033[31m path Failed! \033[0m')
                return

            squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1
            states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

            self.start_state[None, 0:3] = states[0, :3]
            self.end_state[None, 0:3] = states[-1, :3]

            self.states = states.clone().detach().requires_grad_(True)
            self.bp_states = states.clone().detach().requires_grad_(True)

            self.save_poses(self.basefolder / "init_poses" / (str(f"a_star_path")+".json"))

            

            
            if self.dh_collision_info(tensor = states[:, :3], num_inp = 5):
                print(f'Path Colision!!!')
            


            print('\033[31m \033[43m'+f'path'+ '\033[0m' + f', A* IDX:{start, end}, Start --> End Pos:{states[0, :3], states[-1, :3]}, waypoints:{len(path)}')
            print('\033[31m \033[43m'+f'path'+ '\033[0m' + f', A* path: {path}')
            # nerf_loss = self.dh_get_cost()
            # print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')
    

    def dh_learn_init(self):
        # opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)
        # opt = torch.optim.SGD(self.params(), lr=self.lr)
        try:
            condition_met = False
            
            loop = 0
            # 임계값 설정
            threshold = 0.3
            nerf_loss = self.dh_get_cost()
            indices = torch.where(nerf_loss >= threshold)[0]

            self.indices = indices # json save 시 영향
            
            # start waypoint 저장
            if hasattr(self, "basefolder"):
                self.save_poses(self.basefolder / "init_poses" / (str("start")+".json"))
                self.save_costs(self.basefolder / "init_costs" / (str("start")+".json"))
            else:
                print("Warning: data not saved!")
            
            

                ####### grad 방향 제어(xy평면 일반화 상태) ######## 
            self.start_pos = np.array(self.opt.start_pos)
            self.end_pos = np.array(self.opt.end_pos)            
            # 방향 벡터 계산
            #direction_vector = self.end_pos - self.start_pos
            direction_vector = self.end_pos - self.start_pos
            direction_vector /= np.linalg.norm(direction_vector)
            direction_vector = torch.tensor(direction_vector, device='cuda', dtype=torch.float32)

            # 수직인 방향 벡터 생성
            perpendicular_vector = torch.tensor([direction_vector[1], -direction_vector[0], 0.0], device='cuda', dtype=torch.float32)
            perpendicular_vector /= torch.norm(perpendicular_vector)

            
            ##-------- tuning 시 방향비례 세팅 --------##
            # 크기 비교 후 조건에 따라 값 수정
            if abs(perpendicular_vector[0]) / abs(perpendicular_vector[1]) > 5 or abs(perpendicular_vector[1]) / abs(perpendicular_vector[0]) > 5:
                # 극단적으로 크기 차이가 나는 경우 작은 값을 0으로 대체
                if abs(perpendicular_vector[0]) < abs(perpendicular_vector[1]):
                    perpendicular_vector[0] = 0.0
                else:
                    perpendicular_vector[1] = 0.0

            
            dirvec = []
            perpvec = []
            for i in range(self.states[:, :3].shape[0] - 1):
                d = self.states[:, :3][i+1] - self.states[:, :3][i]
                p = torch.tensor([d[1], -d[0], 0]) #+ -가 반시계, - +가 시계
                if abs(p[0]) / abs(p[1]) > 5 or abs(p[1]) / abs(p[0]) > 5:
                    if abs(p[0]) < abs(p[1]):
                        p[0] = 0.0
                    else:
                        p[1] = 0.0
                dirvec.append( d )
                perpvec.append( p )
                
                ####### direction 판단 ######## 
            # directions = {} # index에 대한 딕셔너리로 호출용이
            directions = {}
            directions = self.dh_direction_v2(indices, direction_vector, perpendicular_vector, dirvec, perpvec)
            
            print(directions)
            # Grouping & 다수의 판단따르기 (특수 : 2개의 경우 False로)
            directions, adjacent_groups = self.dh_direction_tuning(directions)
            print(directions)
            # directions[0], directions[1], directions[2], directions[3] = False, False, False, False
            # directions[10], directions[11], directions[12], directions[13] = True, True, True, True
            pdb.set_trace()
            while not condition_met:
                opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)
                for it in range(self.epochs_init):
                    
                    opt.zero_grad()
                    self.epoch = it

                    nerf_loss = self.dh_get_cost()
                    print(loop, it, nerf_loss)
                    
                    
                    # 조건을 기반으로 텐서 값 업데이트
                    indices = torch.where(nerf_loss >= threshold)[0]
                    condition_met = (len(indices) == 0)
                    if condition_met == True:
                        break
                    
                    print(f"target waypoint : {indices}")
                    for i in indices:
                        nerf_loss[i].backward(retain_graph=True)

                    ##-- 전체 s/e의 벡터를 방향벡터로 삼기 --##
                    for i in range(len(indices)):
                        directions_list = [directions.get(i) for i in indices.tolist()]
                        if directions_list[i]: # direct이 true(수직벡터쪽으로) 
                            self.states.grad[indices[i], :3] = perpendicular_vector  * -1 #최종반영은 -grad 이므로
                        else:
                            self.states.grad[indices[i], :3] = perpendicular_vector  * 1

                    # ##-- 바로 앞의 벡터를 방향벡터로 삼아 방향판단 후 진행 --##
                    # for i in range(len(indices)):
                    #     directions_list = [directions.get(i) for i in indices.tolist()]
                    #     if directions_list[i]: # direct이 true(수직벡터쪽으로)
                    #         self.states.grad[indices[i], :3] = perpvec[i]
                    #     else:
                    #         self.states.grad[indices[i], :3] = perpvec[i] * -1


                    ##-- 바로 앞의 벡터를 방향벡터로 삼기(방향판단하지않고 수동으로 제어) --##
                    # for i in indices.tolist():
                    #     self.states.grad[i, :3] = perpvec[i] # 기본 진행방향의 오른쪽(True값), -1 : 왼쪽(False값)
                        # if i == 6 or i == 7 or i == 8:
                        #     self.states.grad[i, :3] = perpvec[i] * -1
                        # else:
                        #     self.states.grad[i, :3] = perpvec[i]
                        # if i == 27: #
                        #     self.states.grad[i, :3] = perpvec[i]  * -1 
                        # else:
                        #     self.states.grad[i, :3] = perpvec[i]
                        # if i == 23: #A*가 조밀할때 급격히 꺾이는 부분에서 방향이 이상해지는 경우가 존재
                        #     self.states.grad[i, :3] = perpvec[i-1]
                    ## 다양한 식을 사용한 결과 다음과 같이 간단하게 방향제어를 하는 것이 유리(0 아니면 크기에 관련없기때문)
                    # self.states.grad[indices, :3] = perpendicular_vector 
                    #self.states.grad[indices, :3] = torch.tensor([0, 0.5, 0])# direction_vector -> FRL path2
                    # self.states.grad[indices, :3] = torch.tensor([0.5, 0, 0]) # rep apt2 path3 case
                    opt.step()


                    """ ####### grad 방향 제어(xy평면 일반화 상태) ######## 

                    ## 기존방법 1
                    self.states.grad[indices, 2] = 0 # z축 계산결과는 0으로 초기화
                    # perpendicular_vector와의 내적 계산하여 대입
                    dot_product = torch.sum(self.states.grad[indices, :3] * perpendicular_vector, dim=1)
                    self.states.grad[indices, :3] = dot_product.view(-1, 1) * perpendicular_vector.view(1, -1)

                    ## 시도방법2 - 이대로 해도 원본이랑 동일한 문제
                    dot_product = torch.sum(self.states.grad[indices, :3] * perpendicular_vector, dim=1)
                    self.states.grad[indices, :3] = perpendicular_vector * torch.norm(self.states.grad[indices, :3], dim=1).view(-1, 1) 

                    ## 시도방법 3 - float64
                    pvec = perpendicular_vector.to(torch.float64)
                    grad = self.states.grad[indices, :3].to(torch.float64)
                    p_grad = pvec * torch.norm(grad, dim=1).view(-1, 1) 
                    cos_grad = torch.dot(pvec, p_grad[1]) / (torch.norm(pvec) * torch.norm(p_grad[1]))
                    self.states.grad[indices, :3] = ( perpendicular_vector.to(torch.float64) * torch.norm(self.states.grad[indices, :3].to(torch.float64), dim=1).view(-1, 1) ).to(torch.float32)

                    ## 방향이 p_vec과 얼마나 차이나는지
                    for i in indices.tolist():
                        cos_grad = torch.dot(perpendicular_vector.to(torch.float64), self.states.grad[i, :3].to(torch.float64)) / (torch.norm(perpendicular_vector.to(torch.float64)) * torch.norm(self.states.grad[i, :3].to(torch.float64)))
                        
                        if cos_grad.item() < 0.98:
                            pdb.set_trace()
                            print('1')
                    """

                    
                    
                    """ 출력
                    print(f"per:{perpendicular_vector}")
                    print(f"previous:{self.states.grad[indices, :3]}")
                    indices = indices.tolist()
                    directions_list = [directions.get(i, False) for i in range(8)]
                    """
                    
                    """ 방향 판단한대로 opt
                    for i in range(len(indices)):
                        directions_list = [directions.get(i) for i in indices.tolist()]
                        
                        # is_left = directions[idx]  # 저장된 결과 확인
                        dot_product_ins = dot_product[i]
                        if directions_list[i]: # direct이 true(수직벡터쪽으로)
                            # print(f"{idx} : direct")
                            if dot_product_ins > 0: # > : perp 동일, < : perp 반대
                                self.states.grad[indices[i], :3] *= -1
                        else:
                            # print(f"{idx} : not direct")
                            if dot_product_ins < 0: # > : perp 동일, < : perp 반대
                                self.states.grad[indices[i], :3] *= -1
                    """ 
                    
                    """ 방향판단하지않고 opt
                    for i in range(len(indices)):
                        dot_product_ins = dot_product[i]
                        if dot_product_ins < 0: # > : perp 동일, < : perp 반대
                            self.states.grad[indices[i], :3] *= -1
                    """
                    
                        
                    """ 연속된 waypoint에 대해서는 같은 방향으로 튜닝이되도록 grad를 조절해야함
                    # self.states.grad[indices] = -self.states.grad[indices] # 이방식은 제어어려워서 폐기
                    # 방법1. 방향 시나리오를 먼저 정해서 if로 부호를 제한
                    for i in indices:
                        if self.states.grad[indices, 0] < 0:
                            self.states.grad[indices, 0] *= -1
                    # 방법2. 인접한 waypoint를 주시대상으로, 먼저 끝난 놈에게 맞게 if로 부호를 동일하게 -> 먼저끝난놈의 순서예측 어려움, 부호를 추적(self.state 해당행의 변화를 관찰)
                    """
                    
                    
                    """
                    count = 0
                    max_attempts = 10 # self.epochs_init으로 대체
                    condition_met = False
                    condition_met = len(indices) 
                    while not len(indices) == 0:
                        # 특정 작업 수행
                        for i in epoch:
                            # 작업수행

                        
                                
                        if some_condition:  # 특정 조건을 충족한 경우 -> indices == 0
                            condition_met = True
                        else:
                            count += 1
                            if count >= max_attempts:
                                count = 0  # 카운트 리셋
                                # 계속해서 루프를 돌릴 작업 수행
                    """

                    save_step = 50
                    if it%save_step == 0:
                        if hasattr(self, "basefolder"):
                            self.save_poses(self.basefolder / "init_poses" / (str(loop+1)+str(it//save_step)+".json"))
                            self.save_costs(self.basefolder / "init_costs" / (str(loop+1)+str(it//save_step)+".json"))
                        else:
                            print("Warning: data not saved!")

                    
                loop = loop + 1
                if condition_met == True:
                    
                    if hasattr(self, "basefolder"):
                        print(f'======= Original Path Tuning =======')
                        self.save_poses(self.basefolder / "init_poses" / (str("end")+".json"))
                        self.save_costs(self.basefolder / "init_costs" / (str("end")+".json"))
                    else:
                        print("Warning: data not saved!")
                    break

        except KeyboardInterrupt:
            print("finishing early")
            if hasattr(self, "basefolder"):
                self.save_poses(self.basefolder / "init_poses" / (str("end")+".json"))
                self.save_costs(self.basefolder / "init_costs" / (str("end")+".json"))
            else:
                print("Warning: data not saved!")

        # # scannet path2 점하나 보정용도 (위에서 방향판단도 꺼야함)
        # self.states = self.states.detach()
        # self.states[5, :2] = torch.tensor([0.4, 0.16])
         
        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        # prev_smooth = torch.cat([self.states[0,None, :], self.states[:-1,:]],        dim=0)
        # next_smooth = to.cat([self.states[1:,:],      self.states[-1,None, :], ], dim=0)
        # self.states = (prev_smooth + next_smooth + self.states)/3
        # self.states = (prev_smooth + self.states)/2
        # self.states = (next_smooth + self.states)/2

        ####################### ##########################################
        states = self.states.clone().detach()
        tuning_posi, _, _, rot_matrix, _, _, _ = self.calc_everything()
        bm = torch.logical_not(self.hybrid_sampling_states == self.states)
        bm_sum = bm.sum(dim=1)
        bm_idx = torch.nonzero( (bm_sum!=0)).reshape(-1)
        
        self.states.requires_grad_(False)
        pdb.set_trace()
        self.states[bm_idx] = self.states[bm_idx] * 1.01 
        print(f'======= Path Tuning (1.01) =======')
        self.save_poses(self.basefolder / "init_poses" / (str("end_1p01")+".json"))

        self.states[bm_idx] = self.states[bm_idx] / 1.01 
        self.states[bm_idx] = self.states[bm_idx] * 1.09 
        print(f'======= Path Tuning (1.09) =======')
        self.save_poses(self.basefolder / "init_poses" / (str("end_1p09")+".json"))

        self.states[bm_idx] = self.states[bm_idx] / 1.09
        self.states[bm_idx] = self.states[bm_idx] * 1.18 
        print(f'======= Path Tuning (1.18) =======')
        self.save_poses(self.basefolder / "init_poses" / (str("end_1p18")+".json"))

        pdb.set_trace()
        ####################### ##########################################

    def dh_learn_init2(self):
        try:
            coods, sigma, coods_BP, num_ways = self.get_path_info()
            coods_z_proj, sigma_edge_idx, sigma_z_proj_all = self.get_bitmapInWay(coods, sigma, coods_BP, num_ways)
            # self.test_BP_approx(coods, sigma, coods_BP, num_ways, coods_z_proj, sigma_edge_idx)
            n_vec_mean = self.get_edge_nvec(coods_z_proj, sigma_z_proj_all, num_ways, sigma_edge_idx)
            
            condition_met = False
            loop = 0
            threshold = 0.3
            while not condition_met:
                opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)
                for it in range(self.epochs_init):
                    
                    opt.zero_grad()
                    self.epoch = it

                    nerf_loss = self.dh_get_cost()
                    print(loop, it, nerf_loss)
                    
                    indices = torch.where(nerf_loss >= threshold)[0]
                    condition_met = (len(indices) == 0)
                    if condition_met == True:
                        break
                    
                    print(f"target waypoint : {indices}")
                    for i in indices:
                        nerf_loss[i].backward(retain_graph=True)
                        self.states.grad[i, :3] = torch.tensor( n_vec_mean[i] )
                    
                    opt.step()

                    save_step = 50
                    if it%save_step == 0:
                        if hasattr(self, "basefolder"):
                            self.save_poses(self.basefolder / "init_poses" / (str(loop+1)+str(it//save_step)+".json"))
                            self.save_costs(self.basefolder / "init_costs" / (str(loop+1)+str(it//save_step)+".json"))
                        else:
                            print("Warning: data not saved!")

                loop = loop + 1
                if condition_met == True:
                    
                    if hasattr(self, "basefolder"):
                        self.save_poses(self.basefolder / "init_poses" / (str("end")+".json"))
                        self.save_costs(self.basefolder / "init_costs" / (str("end")+".json"))
                    else:
                        print("Warning: data not saved!")
                    break

        except KeyboardInterrupt:
            print("finishing early")
            if hasattr(self, "basefolder"):
                self.save_poses(self.basefolder / "init_poses" / (str("end")+".json"))
                self.save_costs(self.basefolder / "init_costs" / (str("end")+".json"))
            else:
                print("Warning: data not saved!")

        
        # smooth path (diagram of which states are averaged)
        # 1 2 3 4 5 6 7
        # 1 1 2 3 4 5 6
        # 2 3 4 5 6 7 7
        # prev_smooth = torch.cat([self.states[0,None, :], self.states[:-1,:]],        dim=0)
        # next_smooth = torch.cat([self.states[1:,:],      self.states[-1,None, :], ], dim=0)
        # self.states = (prev_smooth + next_smooth + self.states)/3
        # self.states = (prev_smooth + self.states)/2
        # self.states = (next_smooth + self.states)/2

        ####################### ##########################################
        pdb.set_trace()
        ####################### ##########################################





    def a_star_init_random_v0(self):
        
        side = self.opt.a_star_grid*5 #PARAM grid size 

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
        output = self.nerf(coods)
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)
        
        # 20, 20, 20
        occupied_value = maxpool(output[None,None,...])[0,0,...]
        occupied = occupied_value > 0.3

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )


        z_bound = self.sr_set_zbound(path = self.opt.path[5:len(self.opt.path)])

        for idx in range(self.opt.total_path_num):
            if self.opt.random_path:
                #print(f'Randomly Select Start/End Points ')    
                #Stonehenge 
                if self.opt.path[5:len(self.opt.path)] == "stonehenge":
                    z_bound = 14
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
    
                #Scannet
                if self.opt.path[5:len(self.opt.path)] == "scannetroom":
                    z_bound = 16
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, z_bound)])
                
                #Scannet 472
                if self.opt.path[5:len(self.opt.path)] == "scannet_472":
                    z_bound = 14
                    start_pos = torch.tensor([np.random.uniform(7, 14), np.random.uniform(1, 19), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(7, 14), np.random.uniform(1, 19), np.random.uniform(10, z_bound)])
    
                #Replica room2
                if self.opt.path[5:len(self.opt.path)] == "scala_room":
                    z_bound = 14
                    start_pos = torch.tensor([np.random.uniform(2, 18), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(2, 18), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
    
                #FRL apt 
                if self.opt.path[5:len(self.opt.path)] == "replica_FRL_v2":
                    z_bound = 15
                    start_pos = torch.tensor([np.random.uniform(5, 15), np.random.uniform(1, 20), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(5, 15), np.random.uniform(1, 20), np.random.uniform(10, z_bound)])
    
                #Replica office3
                if self.opt.path[5:len(self.opt.path)] == "replica_office3":
                    z_bound = 16
                    start_pos = torch.tensor([np.random.uniform(1, 19), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                    end_pos = torch.tensor([np.random.uniform(1, 19), np.random.uniform(6, 15), np.random.uniform(10, z_bound)])
                
                #Replica apt2     
                if self.opt.path[5:len(self.opt.path)] == "replica_apt2":
                    z_bound = 15
                    start_pos = torch.tensor([np.random.uniform(3, 19), np.random.uniform(0, 19), np.random.uniform(10, z_bound)])  #* side / 100
                    end_pos = torch.tensor([np.random.uniform(3, 19), np.random.uniform(0, 19), np.random.uniform(10, z_bound)])  #* side / 100
                
                #Replica apt1    
                if self.opt.path[5:len(self.opt.path)] == "replica_apt1":
                    z_bound = 15
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(3, 16), np.random.uniform(10, z_bound)])  #* side / 100
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(3, 16), np.random.uniform(10, z_bound)])  #* side / 100
    
                #Replica hotel 
                if self.opt.path[5:len(self.opt.path)] == "replica_hotel":
                    z_bound = 16
                    start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(7, 15), np.random.uniform(10, z_bound)])  #* side / 100
                    end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(7, 15), np.random.uniform(10, z_bound)])  #* side / 100
                
                if side == 200:
                    # Start/End Point Matching on 20 -> 40 Grid
                    start = tuple(int(start_pos[i])  for i in range(3) ) 
                    end = tuple(int(end_pos[i])  for i in range(3) ) 
                    start = tuple(x * 2 for x in start)  
                    end = tuple(x * 2 for x in end)  
                else:
                    start = tuple(int(start_pos[i] * side / 100 )  for i in range(3) ) 
                    end = tuple(int(end_pos[i] * side / 100)  for i in range(3) ) 

                
            elif not self.opt.random_path:
                # Set Start/End Points Manually
                print(f'Set Start/End Points by Arguments')    
                start_pos = torch.tensor( [self.opt.start_pos[0] , self.opt.start_pos[1] , self.opt.start_pos[2]] )
                end_pos = torch.tensor( [self.opt.end_pos[0] , self.opt.end_pos[1] , self.opt.end_pos[2]] )
                #convert to index cooredinates
                start_grid_float = grid_size*(start_pos[:3] + 1)/2       # Convert coord to 0~grid_size
                end_grid_float   = grid_size*(end_pos  [:3] + 1)/2
                start = tuple(int(start_grid_float[i]) for i in range(3) )
                end =   tuple(int(end_grid_float[i]  ) for i in range(3) )
                
            self.start_pos = np.array(start_pos.cpu())
            self.end_pos = np.array(end_pos.cpu())
            
            
            if self.opt.path[5:len(self.opt.path)] == "replica_FRL_v2":
                x_max = 10 * side / 100
                y_max = 7 * side / 100  
                if (start[0] < x_max and start[1] < y_max) or (end[0] < x_max and end[1] < y_max): #FRL apt0
                    print(f"path{idx} occupied")
                    continue
            
            # if self.opt.path[5:len(self.opt.path)] == "replica_apt2": #주석이유 : 방내에만 한정하려면 for문 800횟수이상씩 돌아야해서
            #     x_max = 12 * side / 100
            #     y_min = 3 * side / 100  
            #     if (start[0] < x_max and start[1] > y_min) or (end[0] < x_max and end[1] > y_min): #rep apt2
            #         print(f"path{idx} occupied")
            #         continue

            if self.opt.path[5:len(self.opt.path)] == "scannet_472":
                x_max = 8 * side / 100
                y_max = 9 * side / 100  
                if (start[0] < x_max and start[1] < y_max) or (end[0] < x_max and end[1] < y_max): #scannet 472
                    print(f"path{idx} occupied")
                    continue

            if occupied[start]:
                print(f"path{idx} start {start} waypoint occupied !!!")
                continue
            elif occupied[end]:
                print(f"path{idx} end {end} waypoint occupied !!!")
                continue
            
            path = astar(occupied, start, end, self.nerf, self.robot_body, side, z_bound, self.opt.path[5:len(self.opt.path)-1], self.opt)
            
            if path == None:
                print(f'\033[31m path{idx} S/E colision! \033[0m')
                continue
            elif path == 0:
                print(f'\033[31m path{idx} Failed! \033[0m')
                continue
 
            ##---------- 3) path save -------------##
            squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1
            states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

            self.start_state[None, 0:3] = states[0, :3]
            self.end_state[None, 0:3] = states[-1, :3]

            #path info
            print('\033[31m \033[43m'+f'path{idx}'+ '\033[0m' + f', A* IDX:{start, end}, Start --> End Pos:{states[0, :3], states[-1, :3]}, waypoints:{len(path)}')
            print('\033[31m \033[43m'+f'path{idx}'+ '\033[0m' + f', A* path: {path}')

            ##---------- 4) smooth path (diagram of which states are averaged) -------------## -> A* 에선 필요없음
            # 1 2 3 4 5 6 7
            # 1 1 2 3 4 5 6
            # 2 3 4 5 6 7 7
            #prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
            #next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
            #states = (prev_smooth + next_smooth + states)/3
 
            ### Set A Star Wayp Coordinate ###
            self.states = states.clone().detach().requires_grad_(True)
            self.bp_states = states.clone().detach().requires_grad_(True)

            self.save_poses(self.basefolder / "init_poses" / (str(f"a_star_path{idx}")+".json"))
            
            # A* path colision info
            nerf_loss = self.dh_get_cost()
            print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')



    def dh_calc_everything(self):
        
        start_pos   = self.start_state[None, 0:3]
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


        ####################### New Pos, Rot ##########################################
        # start_pos_dh = torch.tensor(self.start_pos)
        # end_pos_dh = torch.tensor(self.end_pos)

        # path 0, 1 모두 사용
        current_pos_dh = torch.cat( [start_pos, start_pos, self.states[:, :3], end_pos], dim=0)

        prev_pos_dh = current_pos_dh[:-1, :]
        next_pos_dh = current_pos_dh[1: , :]

        current_vel_dh = (next_pos_dh - prev_pos_dh)/self.dt
        current_vel_dh = torch.cat( [ current_vel_dh, end_v], dim=0)

        prev_vel_dh = current_vel_dh[:-1, :]
        next_vel_dh = current_vel_dh[1: , :]

        current_accel_dh = (next_vel_dh - prev_vel_dh)/self.dt - self.g

        # duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel_dh = torch.cat( [ current_accel_dh, current_accel_dh[-1,None,:] ], dim=0)

        accel_mag_dh     = torch.norm(current_accel_dh, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body_dh = current_accel_dh/accel_mag_dh

        # remove states with rotations already constrained
        z_axis_body_dh = z_axis_body_dh[2:-1, :]

        z_angle_dh = self.states[:,3]

        in_plane_heading_dh = torch.stack( [torch.sin(z_angle_dh), -torch.cos(z_angle_dh), torch.zeros_like(z_angle_dh)], dim=-1)

        x_axis_body_dh = torch.cross(z_axis_body_dh, in_plane_heading_dh, dim=-1)
        x_axis_body_dh = x_axis_body_dh/torch.norm(x_axis_body_dh, dim=-1, keepdim=True)
        y_axis_body_dh = torch.cross(z_axis_body_dh, x_axis_body_dh, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix_dh = torch.stack( [x_axis_body_dh, y_axis_body_dh, z_axis_body_dh], dim=-1)

        ## rot를 path와 동일한 차원으로(+3차원 X, 1:1대응)
        # rot_matrix_dh = torch.cat( [start_R, next_R, rot_matrix_dh, end_R], dim=0)

        ####################### ##########################################

        return current_pos_dh, rot_matrix_dh
    
    def dh_rotation(self, stat):
        
        start_pos   = self.start_state[None, 0:3]
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


        ####################### New Pos, Rot ##########################################
        # start_pos_dh = torch.tensor(self.start_pos)
        # end_pos_dh = torch.tensor(self.end_pos)

        # path 0, 1 모두 사용
        # current_pos_dh = torch.cat( [start_pos, start_pos, self.states[:, :3], end_pos], dim=0)
        current_pos_dh = torch.cat( [stat], dim=0 )

        prev_pos_dh = current_pos_dh[:-1, :]
        next_pos_dh = current_pos_dh[1: , :]

        current_vel_dh = (next_pos_dh - prev_pos_dh)/self.dt
        current_vel_dh = torch.cat( [ current_vel_dh, end_v], dim=0)

        prev_vel_dh = current_vel_dh[:-1, :]
        next_vel_dh = current_vel_dh[1: , :]

        current_accel_dh = (next_vel_dh - prev_vel_dh)/self.dt - self.g

        # duplicate last accceleration - its not actaully used for anything (there is no action at last state)
        current_accel_dh = torch.cat( [ current_accel_dh, current_accel_dh[-1,None,:] ], dim=0)

        accel_mag_dh     = torch.norm(current_accel_dh, dim=-1, keepdim=True)
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ########################################## 
        # needs to be pointing in direction of acceleration
        z_axis_body_dh = current_accel_dh/accel_mag_dh
        # 아래부분 때문에 4,3이 1,3이 되어 에러발생
        # remove states with rotations already constrained
        # z_axis_body_dh = z_axis_body_dh[2:-1, :]
        
        # z_angle_dh = self.states[:,3]
        stat = torch.cat( [stat, torch.zeros( (stat.shape[0], 1) ) ], dim=-1)
        z_angle_dh = stat[:,3]

        in_plane_heading_dh = torch.stack( [torch.sin(z_angle_dh), -torch.cos(z_angle_dh), torch.zeros_like(z_angle_dh)], dim=-1)

        x_axis_body_dh = torch.cross(z_axis_body_dh, in_plane_heading_dh, dim=-1)
        x_axis_body_dh = x_axis_body_dh/torch.norm(x_axis_body_dh, dim=-1, keepdim=True)
        y_axis_body_dh = torch.cross(z_axis_body_dh, x_axis_body_dh, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix_dh = torch.stack( [x_axis_body_dh, y_axis_body_dh, z_axis_body_dh], dim=-1)

        ## rot를 path와 동일한 차원으로(+3차원 X, 1:1대응)
        # rot_matrix_dh = torch.cat( [start_R, next_R, rot_matrix_dh, end_R], dim=0)

        ####################### ##########################################
 
        return current_pos_dh, rot_matrix_dh

    def get_path_info(self):
        # body points (24, 500, 3)
        coods = self.dh_body_to_world(self.robot_body)
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

        return coods, sigma, coods_BP, num_ways 
    def get_bitmapInWay(self, coods, sigma, coods_BP, num_ways):
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
        sigma_edge_idx = [ torch.where( sigma_z_proj_all[way] )[0] for way in range(0, num_ways) ]
        
        return coods_z_proj, sigma_edge_idx, sigma_z_proj_all
    def get_edge_nvec(self, coods_z_proj, sigma_z_proj_all, num_ways, sigma_edge_idx):
        n_vec_mean, midpoint_mean = [], []
            
        for way in range(0, num_ways):
            arr = sigma_z_proj_all[way].reshape(10, 10).numpy().astype(np.uint8) * 255 # T값을 255(W)로 변환
            
            sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)  # x 방향 Sobel 필터
            sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)  # y 방향 Sobel 필터

            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            sigma_z_filter = ( torch.tensor(sobel_combined) > 0.3 ).reshape(100)
            
            sigma_edge_idx[way] = torch.where( sigma_z_proj_all[way] & sigma_z_filter.cpu() )[0]
            
            Layer = 4
            n_vec, midpoint = [], []
            for i in range( len(sigma_edge_idx[way])-1 ):
                grad_vec = coods_z_proj[Layer][way][sigma_edge_idx[way][i+1]] - coods_z_proj[Layer][way][sigma_edge_idx[way][i]]
                n_vec.append( torch.tensor([ -grad_vec[1], grad_vec[0], grad_vec[2] ]) )   # 2D는 (-y,x) : 반시계로 90도 회전 // 3D는 cross(a,b)이지만 이경우는 z사영이므로
                midpoint.append( torch.tensor( (coods_z_proj[Layer][way][sigma_edge_idx[way][i+1]] + coods_z_proj[Layer][way][sigma_edge_idx[way][i]]) / 2.0 , dtype=torch.float32, requires_grad=True).to('cuda') )
                
            
            if n_vec and midpoint:
                n_vec = torch.stack( n_vec ).cpu().detach().numpy()
                midpoint = torch.stack( midpoint ).cpu().detach().numpy()
            else:
                n_vec = torch.tensor( n_vec )
                midpoint = torch.tensor( midpoint )
            n_vec_mean.append( n_vec.mean(0) )
            midpoint_mean.append( midpoint.mean(0) )

            # fig, ax1 = plt.subplots(1, 1, subplot_kw={"projection":"3d"})
            # for i in range(len(midpoint)):
            #     ax1.quiver(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2],
            #                 n_vec[i, 0], n_vec[i, 1], n_vec[i, 2], color='green')
            #     ax1.quiver(midpoint.mean(0)[0], midpoint.mean(0)[1], midpoint.mean(0)[2],
            #                 n_vec.mean(0)[0], n_vec.mean(0)[1], n_vec.mean(0)[2], color='cyan')
            #     ax1.set_title(f"way{way}")
        return n_vec_mean
    
    
    def dh_get_cost(self):
        density = self.nerf( self.dh_body_to_world(self.robot_body) )**2        
        colision_prob = torch.mean(density, dim = -1)                           
        return colision_prob                                                    
    
    def dh_body_to_world(self, robot_body, states=None):
        
        if states is None:
            states = self.states

        num_matrices = states.size(0)      # waypoints num 
        identity_matrices = torch.eye(3).unsqueeze(0).expand(num_matrices, -1, -1) # expand identity matrix to way points num (batch) 
        self_states_tensor = states[:, :3] # waypoints x,y,z

        # world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################        
        
        #pos, rot_matrix_dh = self.dh_calc_everything()

        world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]

        return world_points.swapdims(-1,-2)     # waypoints, bodypoints, 3(xyz)
    
    """ 특정 점들에 대한 body density 출력하고 양쪽을 비교하기
    """
    def dh_get_cost_target(self, states):
        # density = self.nerf(self.states[:, :3])

        density = self.nerf( self.dh_body_to_world_target(states, self.robot_body) )**2
        # print(density.shape)
        colision_prob = torch.mean(density, dim = -1) 
        return colision_prob # colision_prob*1e6

    def dh_body_to_world_target(self, states, robot_body):
        # states [ S, 3] xyz 좌표가 담긴 정보들
        num_matrices = states.size(0)  
        identity_matrices = torch.eye(3).unsqueeze(0).expand(num_matrices, -1, -1)  
        self_states_tensor = states[:, :]
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################   
        # world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]
        pos, rot_matrix_dh = self.dh_rotation(states)
        # pos, rot_matrix_dh = self.dh_calc_everything() # 이거 그대로쓰면 차원에러남

        world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]

        return world_points.swapdims(-1,-2) 

    def dh_direction(self, indices, direction_vector, perpendicular_vector, dirvec, perpvec):
        # occupied 점 양쪽(pvec)과 양위쪽(pvec+dvec)의 density를 비교
        directions = {} # index에 대한 딕셔너리로 호출용이
        indices = indices.tolist()  # 텐서를 리스트로 해야 i가 숫자로 들어가면서 dict를 숫자로 호출가능
        # directions = []
        for i in indices:
        # for i in range(len(indices)):
            
            # x,y 한 축 이동
            # targ_tensor_x1 = self.states[i, :3].clone()
            # targ_tensor_x1[0] += 0.2

            # targ_tensor_x2 = self.states[i, :3].clone() 
            # targ_tensor_x2[0] -= 0.2

            # targ_tensor_y1 = self.states[i, :3].clone()
            # targ_tensor_y1[1] += 0.2

            # targ_tensor_y2 = self.states[i, :3].clone() 
            # targ_tensor_y2[1] -= 0.2

            # target_states = torch.stack([targ_tensor_x1, targ_tensor_x2, targ_tensor_y1, targ_tensor_y2])   
            
            # 방향, 수직벡터로 이동
            targ_tensor_p1 = self.states[i, :3].clone()
            targ_tensor_p1 += 0.1 * perpvec[i] # perpendicular_vector

            targ_tensor_p1d1 = self.states[i, :3].clone()
            targ_tensor_p1d1 = targ_tensor_p1 + 0.1 * dirvec[i] #direction_vector

            targ_tensor_p2 = self.states[i, :3].clone() 
            targ_tensor_p2 -= 0.1 * perpvec[i] #perpendicular_vector

            targ_tensor_p2d2 = self.states[i, :3].clone() 
            targ_tensor_p2d2 = targ_tensor_p2 + 0.1 * dirvec[i] #direction_vector

            target_states2 = torch.stack([targ_tensor_p1, targ_tensor_p1d1, targ_tensor_p2, targ_tensor_p2d2]) 

            # new_x1 = x + 0.2
            # new_y1 = y
            # new_z1 = z
            
            # new_x2 = x - 0.2
            # new_y2 = y
            # new_z2 = z
            
            # new_x3 = x
            # new_y3 = y + 0.2
            # new_z3 = z
            
            # new_x4 = x
            # new_y4 = y - 0.2
            # new_z4 = z

            # target_states.extend([
            #     [new_x1, new_y1, new_z1], # +x
            #     [new_x2, new_y2, new_z2], # -x
            #     [new_x3, new_y3, new_z3], # +y
            #     [new_x4, new_y4, new_z4]  # -y
            # ])
            
            # target_states = torch.tensor(target_states)
            target_loss = self.dh_get_cost_target(target_states2)

            loss_averaged = target_loss.view(-1, 2).mean(dim=1)
            if loss_averaged[0] < loss_averaged[1]: # +-X(-> direction)의 loss가 더 작다면 
                directions[i] = True # 방향을 direction으로
            else: # +-Y의 loss가 더 크다면
                directions[i] = False # 반대로
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################
        return directions
    
    def dh_direction_v2(self, indices, direction_vector, perpendicular_vector, dirvec, perpvec):
        # occupied 점 양쪽(pvec)의 각 4개 점 density를 서로 비교
        directions = {} # index에 대한 딕셔너리로 호출용이
        indices = indices.tolist()  # 텐서를 리스트로 해야 i가 숫자로 들어가면서 dict를 숫자로 호출가능
        # directions = []
        for i in indices:
            # 방향, 수직벡터로 이동
            mv_dist = 0.1

            targ_tensor_p1 = self.states[i, :3].clone()
            targ_tensor_p1 += mv_dist * perpendicular_vector # perpvec[i] # perpendicular_vector
            targ_tensor_p12 = targ_tensor_p1 + mv_dist * perpendicular_vector
            targ_tensor_p123 = targ_tensor_p12 + mv_dist * perpendicular_vector
            targ_tensor_p1234 = targ_tensor_p123 + mv_dist * perpendicular_vector

            targ_tensor_i_p1 = self.states[i, :3].clone()
            targ_tensor_i_p1 -= mv_dist * perpendicular_vector # perpvec[i] # perpendicular_vector
            targ_tensor_i_p12 = targ_tensor_i_p1 - mv_dist * perpendicular_vector
            targ_tensor_i_p123 = targ_tensor_i_p12 - mv_dist * perpendicular_vector
            targ_tensor_i_p1234 = targ_tensor_i_p123 - mv_dist * perpendicular_vector
            target_states2 = torch.stack([targ_tensor_p1, targ_tensor_p12, targ_tensor_p123, targ_tensor_p1234, 
                                          targ_tensor_i_p1, targ_tensor_i_p12, targ_tensor_i_p123, targ_tensor_i_p1234]) 

            # target_states = torch.tensor(target_states)
            target_loss = self.dh_get_cost_target(target_states2)
            loss_averaged = target_loss.view(-1, int(target_states2.shape[0]/2)).mean(dim=1)

            pvec_sum = torch.sum( ( target_loss.view(-1, int(target_states2.shape[0]/2))[0] < 0.3 ) == True )
            i_pvec_sum = torch.sum( ( target_loss.view(-1, int(target_states2.shape[0]/2))[1] < 0.3 ) == True )
            if pvec_sum == i_pvec_sum:
                
                if loss_averaged[0] < loss_averaged[1]: # +-X(-> direction)의 loss가 더 작다면 
                    directions[i] = True # 방향을 direction으로
                else: # +-Y의 loss가 더 크다면
                    directions[i] = False # 반대로
            else: 
                if pvec_sum > i_pvec_sum: #빈공간 갯수가 많은쪽이 pvec
                    directions[i] = True
                else:
                    directions[i] = False

            
            print(f'direct target : {target_states2}')
            print(f'inference : {target_loss}')
            print(f'avg : {loss_averaged}')
            # ####################### ##########################################
            # pdb.set_trace()
            # ####################### ##########################################
        print(f'directions: {directions}')
        return directions

    def dh_direction_tuning(self, direct):
        
        adjacent_groups = []
        current_group = []

        for num in sorted(direct.keys()):
            if not current_group or num == current_group[-1] + 1: 
                current_group.append(num)
            else:  
                adjacent_groups.append(current_group)
                current_group = [num]
        if current_group: 
            adjacent_groups.append(current_group)

        for group in adjacent_groups:
            values = [direct[num] for num in group]
            majority_value = max(set(values), key=values.count) 
            for num in group:
                direct[num] = majority_value
        
        return direct, adjacent_groups
    
    def dh_Encoding_mem_req(self, bitmap_tensor, bitmap_threshold):
        import math
        import pandas as pd
        from scipy.sparse import csr_matrix

        def run_length_encoding(array):
            """
            Convert a sequence of 0s and 1s to its run-length encoded format.

            Args:
                array (list): The input array of 0s and 1s to be encoded.

            Returns:
                tuple: Tuple containing two lists - count and data.

            Examples:
                >>> run_length_encoding([1, 1, 0, 0, 0, 1, 1, 1])
                ([2, 3], [1, 0])
            """
            count = []
            data = []

            if not array:
                return count, data

            count.append(1)
            data.append(array[0])

            # Iterate through the elements in the array
            for i in range(1, len(array)):
                # If the current element is the same as the previous element, increase the count
                if array[i] == array[i - 1]:
                    count[-1] += 1
                else:
                    # If the current element is different from the previous element, add the count and element to the lists
                    count.append(1)
                    data.append(array[i])

            return count, data
        
        count, data = run_length_encoding( bitmap_tensor.cpu().numpy().reshape(-1).astype(int).tolist() )
        count_arr = np.array(count)
        data_arr = np.array(data)

        print(f'Threshold : {bitmap_threshold}')
        print(f'(RLE Length) count: {count_arr.shape}, data: {data_arr.shape}')
        print(f'(RLE Max)    count: {np.max(count_arr)}, data: {np.max(data_arr)}')
        count_bits = count_arr.shape[0] * math.ceil(math.log2( np.max(count_arr) ))
        data_bits = data_arr.shape[0] * np.max(data_arr)
        print(f'C: {count_bits} + D: {data_bits} = {count_bits + data_bits} bits')


        pd.DataFrame(data=count_arr).to_csv("data/RLE_Count.csv", index=False, header=False)
        pd.DataFrame(data=data_arr).to_csv("data/RLE_Data.csv", index=False, header=False)


        print('#####---------- CSR Compression ----------#####')
        # z-ax 3D -> 2D grid
        occupied_Origin_array_2D = bitmap_tensor.cpu().numpy().reshape(-1, bitmap_tensor.cpu().numpy().shape[-1] )
        sparse_matrix = csr_matrix( occupied_Origin_array_2D.astype(int) )

        # 3요소 추출
        indptr = sparse_matrix.indptr   # Row pointer
        indices = sparse_matrix.indices # Col
        data_values = sparse_matrix.data

        # csv 저장
        # print(f'CSR Matrix : {sparse_matrix}')
        print(f'Threshold : {bitmap_threshold}')
        print(f'(CSR Length) Row ptr : {indptr.shape}, Col : {indices.shape}, Data : {data_values.shape}')
        print(f'(CSR Max) Row ptr : {np.max(indptr)}, Col : {np.max(indices)}, Data : {np.max(data_values)}')

        # 필요 bit 계산 - length * ceil(log2(max)) ⇒ XX bit + YY bit + ZZ bit = Total
        
        indptr_bits = indptr.shape[0] * math.ceil(math.log2( np.max(indptr) ))
        indices_bits = indices.shape[0] * math.ceil(math.log2( np.max(indices) ))
        data_values_bits = data_values.shape[0] * 1 # math.ceil(math.log2( np.max(data_values) ))

        req_bits = indptr_bits + indices_bits + data_values_bits
        
        print(f'R: {indptr_bits} + C: {indices_bits} + D: {data_values_bits} = {req_bits} bits')

        pd.DataFrame(data=indptr).to_csv("data/CSR_Row_ptr.csv", index=False, header=False)
        pd.DataFrame(data=indices).to_csv("data/CSR_Col.csv", index=False, header=False)
        pd.DataFrame(data=data_values).to_csv("data/CSR_Data.csv", index=False, header=False)
        return
    def dh_MinPool3d(self, in_tensor, kernel_size=5, stride=5):
        input_tensor = in_tensor.reshape(1, 1, 100, 100, 100)  # (batch_size, channels, depth, height, width)

        # 출력 텐서 크기 계산
        output_depth = (input_tensor.shape[2] - kernel_size) // stride + 1
        output_height = (input_tensor.shape[3] - kernel_size) // stride + 1
        output_width = (input_tensor.shape[4] - kernel_size) // stride + 1
        # 출력 텐서를 저장할 빈 텐서 생성
        output_tensor = torch.zeros(1, 1, output_depth, output_height, output_width)

        # MaxPooling 연산 수행
        for d in range(output_depth):
            for h in range(output_height):
                for w in range(output_width):
                    # 커널 내의 최댓값 찾기
                    window = input_tensor[:, :, d * stride:d * stride + kernel_size,
                                        h * stride:h * stride + kernel_size,
                                        w * stride:w * stride + kernel_size]
                    output_tensor[:, :, d, h, w] = torch.min(window)

        return output_tensor[0, 0, ...]


    def dh_collision_info(self, tensor, num_inp):
        def linear_interpolation(point1, point2, steps):
            return [point1 + i * (point2 - point1) / (steps - 1) for i in range(steps - 1)]
        
        sublist = []
        for i in range(len(tensor) - 1):
            sublist.append(torch.stack(linear_interpolation(tensor[i], tensor[i + 1], num_inp)))

        result_tensor = torch.cat(sublist, dim=0)
        result_tensor = torch.cat( (result_tensor, tensor[-1:, :]), dim=0  )
        
        
        target_loss = self.dh_get_cost_target(result_tensor)

        return (target_loss > 0.3).any()
 


######################################## SR NUM GRAD ESTIMATOR #################################################
    def sr_num_grad_selector(self):

        # get collision way points
        outputs0 = self.sr_get_collision_waypoints()

        # get bmap on collision waypoints
        outputs1 = self.sr_get_collision_bmap( outputs0['collision_way_idx'] )
        
        # get projected bmap on collision waypoints
        outputs2 = self.sr_get_projected_collision_bmap( outputs1['occupancy_bmap'] )

        # get edge
        outputs3 = self.sr_get_edge( outputs2['projected_collision_bmap'] )

        # get nvec of edge
        A = outputs3['filter']
        edge_skip = torch.sum( torch.sum( A, 2 ), 1 )==0
        print(f'Occupied BMap Filter: {A}, Occupied BMap Filter Shape: {A.shape}')
        outputs4 = self.sr_get_nvec( outputs3['filter'] )

        # evaluate bp & nvec of edge
        outputs5 = self.sr_eval_nvec( outputs4['avg_normal_vec'] , outputs4['edge_idx'])

        # Edge Num Grad --> outputs0['avg_normal_vec']
        # Edge BP Grad --> outputs1['edge_bpgrad']

        outputs = {
            'num_grad'                  :   outputs4['avg_normal_vec'],
            'edge_bpgrad'               :   outputs5['edge_bpgrad'],
            'occupancy_bmap'            :   outputs1['occupancy_bmap'],
            'projected_bmap'            :   outputs2['projected_collision_bmap'],
            'edge_cos_sim'              :   outputs5['edge_cos_sim'],
            'global_cos_sim'            :   outputs5['global_cos_sim'],
            'edge_skip'                 :   edge_skip
        }

        return outputs

    def sr_get_collision_waypoints(self):
        # Get bodypoints on waypoints
        body_coods_on_way = self.dh_body_to_world(self.robot_body)          # waypoints, bodypoints, 3(xyz)

        # Get sigma @ bodypoints on waypoints
        sigmas = self.nerf( body_coods_on_way )**2                          # waypoints, bodypoints

        # Get Occupied Bit Map on waypoints
        occupied_bmap = sigmas > self.opt.occupancy_th                      # waypoints, bodypoints

        # Get Occupied Points Number on waypoints
        occupied_cnt = torch.sum( occupied_bmap, 1 )                        # waypoints

        # Get Occupied Ratio on waypoints
        occupied_ratio = occupied_cnt / body_coods_on_way.shape[1] / body_coods_on_way.shape[2]
        self.occupancy = occupied_ratio != 0
        occupied_idx = torch.nonzero( self.occupancy ).reshape(-1)

        outputs = {
            'collision_way_idx': occupied_idx
        }

        return outputs

    def sr_body_to_world(self, robot_body, collision_idx):
        identity_matrices = torch.eye(3).unsqueeze(0).expand(collision_idx.shape[0], -1, -1) # expand identity matrix to way points num (batch) 
        self_states_tensor = self.states[collision_idx, :3] # waypoints x,y,z
        world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]
        return world_points.swapdims(-1,-2)

    def sr_get_collision_bmap(self, collision_idx):
        coods = self.sr_body_to_world( self.robot_body, collision_idx )
        #coods_reshape = coods.reshape( collision_idx.shape[0], self.body_nbins[0], self.body_nbins[1], self.body_nbins[2] , -1)
        #print(f'{coods_reshape.shape}')
        #coods_reshape = coods_reshape.swapdims( 1, 3 )
        #print(f'{coods_reshape.shape}')
        sigmas = self.nerf( coods )**2    # collision waypoints num , bodypoints (500)

        sigmas = sigmas.reshape( collision_idx.shape[0], self.body_nbins[0], self.body_nbins[1], self.body_nbins[2] )
        sigmas = sigmas.swapdims( 1, 3 )            # collision waypoints num, zaxis_nbins, xaxis_nbins, yaxis_nbins

        occupied_bmap = sigmas > self.opt.occupancy_th

        outputs = {
            'occupancy_bmap': occupied_bmap
        }
        
        return outputs

    def sr_get_projected_collision_bmap(self, collision_bmap):
        collision_nwaypoints = collision_bmap.shape[0]

        projected_collision_bmap = collision_bmap[:,0,:,:]
        for z in range (self.body_nbins[2]-1):
            projected_collision_bmap = projected_collision_bmap | collision_bmap[:,z+1,:,:]
        
        outputs = {
            'projected_collision_bmap': projected_collision_bmap
        }

        return outputs

    def sr_get_edge(self, collision_bmap):
        bmap = collision_bmap.detach().cpu().numpy().astype(np.uint8)
        occupied_idx = torch.nonzero( self.occupancy ).reshape(-1)
        filter_result = torch.Tensor()
        #print(f'{self.occupancy}')
        #print(f'{collision_bmap.shape}')
        for wpoints in range (collision_bmap.shape[0]):                             # Collision Waypoints Number
            #print(f'======= Occupancy Bmap =======')
            #print(f'{collision_bmap[wpoints]}')
            img_sobel_x = cv2.Sobel( bmap[wpoints], cv2.CV_64F, 1, 0, ksize=3 )     # Detect Vertical Edge
            img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

            img_sobel_y = cv2.Sobel( bmap[wpoints], cv2.CV_64F, 0, 1, ksize=3 )     # Detect Horizontal Edge
            img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

            sobel_filter = img_sobel_x + img_sobel_y
            sobel_filter = (sobel_filter > 0)
            sobel_filter = torch.tensor( sobel_filter & bmap[wpoints], device="cuda" ).bool()
            #print(f'======= Before Points Removal Sobel Filter =======')
            #print(f'{sobel_filter}')
            
            sobel_filter_idx = torch.nonzero( sobel_filter, as_tuple=True )
            for edge in range (sobel_filter_idx[0].shape[0]):
                # Points Removal
                search_list = []
                if sobel_filter_idx[0][edge] == 0:
                    # y+1 --> row+1
                    search_list.append(torch.tensor([ 1, 0 ], device="cuda"))

                elif sobel_filter_idx[0][edge] == self.body_nbins[0]-1:
                    # y-1 --> row-1
                    search_list.append(torch.tensor([ -1,0 ], device="cuda"))

                elif sobel_filter_idx[0][edge] > 0 and sobel_filter_idx[0][edge] < self.body_nbins[0]-1 :
                    search_list.append(torch.tensor([  1, 0 ], device="cuda"))
                    search_list.append(torch.tensor([ -1, 0 ], device="cuda"))

                if sobel_filter_idx[1][edge] == 0:
                    # x+1 --> col+1
                    search_list.append(torch.tensor([ 0, 1 ], device="cuda"))

                elif sobel_filter_idx[1][edge] == self.body_nbins[1]-1:
                    # x-1 --> col-1
                    search_list.append(torch.tensor([ 0, -1 ], device="cuda"))
                elif sobel_filter_idx[1][edge] > 0 and sobel_filter_idx[1][edge] < self.body_nbins[1]-1:
                    search_list.append(torch.tensor([ 0, 1 ], device="cuda"))
                    search_list.append(torch.tensor([ 0,-1 ], device="cuda"))
                
                for neighbor in range (len(search_list)):
                    neighbor_row = sobel_filter_idx[0][edge] + search_list[neighbor][0]
                    neighbor_col = sobel_filter_idx[1][edge] + search_list[neighbor][1]

                    if neighbor==0: remove_flag = collision_bmap[wpoints][neighbor_row][neighbor_col]
                    else: remove_flag = remove_flag & collision_bmap[wpoints][neighbor_row][neighbor_col]
                
                if remove_flag:
                    sobel_filter[ sobel_filter_idx[0][edge], sobel_filter_idx[1][edge] ] = False
            
            #print(f'======= After Points Removal Sobel Filter =======')
            #print(f'{sobel_filter}')           



            if filter_result.shape[0] == 0:
                filter_result = sobel_filter
            else:
                filter_result = torch.cat( [filter_result, sobel_filter], dim=0 )

            # Update Occupancy Info by Filter Results
            #if not sobel_filter.sum().bool():
            #    # Empty
            #    self.occupancy[ occupied_idx[wpoints] ] = False           
            #    continue


        
        filter_result = filter_result.reshape( -1, bmap.shape[1], bmap.shape[2] )
        
        outputs = {
            'filter': filter_result
        }

        return outputs

    def sr_get_nvec(self, collision_bmap_edge):
        '''
            collision_bmap_edge = Sobel Filter Output
        '''
        
        # get occupancy bmap
        occupancy_bmap = self.occupancy
        occupancy_bmap_idx = torch.nonzero(occupancy_bmap).reshape(-1)

        # get waypoints coordinates
        occupied_way_coords = self.sr_body_to_world( self.robot_body, occupancy_bmap_idx )
        # waypoints, zaxis_nbins, x_axis_nbins(row), y_axis_nbins(col)
        occupied_way_coords = occupied_way_coords.reshape( -1, self.body_nbins[0], self.body_nbins[1], self.body_nbins[2], 3 ).swapdims(1,3)
        
        wayp_normal_vec = list()    # normal vector list @ each waypoint
        avg_normal_vec = torch.Tensor()
        edge_idx_list = list()
        skip_num_grad = torch.Tensor()

        for wpoints in range (collision_bmap_edge.shape[0]):
            # Get bmap edge idx
            
            # Get All False Filter Output --> [0,0] Filter Normal Vector
            if collision_bmap_edge[wpoints].sum() == 0 :
                if avg_normal_vec.shape[0] == 0 : avg_normal_vec = torch.tensor([[0,0]], device="cuda")
                else: avg_normal_vec = torch.cat( [avg_normal_vec, torch.tensor([[0,0]], device="cuda")], dim=0 )
                edge_idx_list.append( torch.Tensor() )
            
                #if skip_num_grad.shape[0] == 0: skip_num_grad = torch.tensor([identical_start_goal], device="cuda").reshape(-1)
                #else: skip_num_grad = torch.cat( [skip_num_grad, torch.tensor([identical_start_goal], device="cuda").reshape(-1)] )
                continue

            edge_idx = torch.nonzero(collision_bmap_edge[wpoints], as_tuple=True)       # edge_idx[0]: row & edge_idx[1]: col
            edge_idx_list.append( edge_idx )
            row = edge_idx[0]
            col = edge_idx[1]
            #print(f'row: {row}')
            #print(f'col: {col}')

            # Get start end points
            idx_list, idx_combs = self.sr_get_start_goal(edge_idx)
            path = -1
            j = -1
            #print(f'idx combs: {idx_combs} , idx shape: {idx_combs.shape}')
            

            if idx_combs.shape[0] == 0 : 
                identical_start_goal = True
            else: 
                identical_start_goal = False

            if identical_start_goal == False:
                while (path==-1):
                    j=j+1

                    if j > idx_list.shape[0]-1: break

                    start = torch.tensor( [ edge_idx[0][idx_combs[idx_list[j]][0]] , edge_idx[1][idx_combs[idx_list[j]][0]] ] )
                    goal  = torch.tensor( [ edge_idx[0][idx_combs[idx_list[j]][1]] , edge_idx[1][idx_combs[idx_list[j]][1]] ] )
                    
                    start = tuple(int(start[i]) for i in range(2) )
                    goal =   tuple(int(goal[i]  ) for i in range(2) )

                    # Get Boundary Points Tuple
                    path = astar_2d(occupied = torch.logical_not(collision_bmap_edge[wpoints]), start=start, goal=goal)
                    if path == -1 :
                        continue
                    #print(f'wpoints{occupancy_bmap_idx[wpoints]} edge: {path}')
               
                if path != -1:
                    normal_vec_list = torch.Tensor()
                    for p in range (len(path)-1):
                        # Get x,y coord
                        start = occupied_way_coords[wpoints][round(self.body_nbins[2]/2)][path[p][0]][path[p][1]][:2]
                        end = occupied_way_coords[wpoints][round(self.body_nbins[2]/2)][path[p+1][0]][path[p+1][1]][:2]

                        edge_vec = start - end
                        edge_vec_normalized = edge_vec / torch.norm( edge_vec )
                        #print(f'norm_edge_vec: {edge_vec_normalized}')

                        # Normal Vector of Edge
                        normal_vec = torch.tensor( [-edge_vec_normalized[1], edge_vec_normalized[0]], device="cuda" )
                        
                        # Concat Normal Vector of Edge
                        if normal_vec_list.shape[0] == 0:
                            normal_vec_list = normal_vec.reshape(1,-1)
                        else:
                            normal_vec_list = torch.cat( [normal_vec_list, normal_vec.reshape(1,-1)], dim=0 )
                    
                    # Average Normal Vector
                    avg_normal_vec_normalized = torch.mean( normal_vec_list, 0 ).reshape(1,-1)

                    # Normalize Average Normal Vector
                    avg_normal_vec_normalized = avg_normal_vec_normalized / torch.norm( avg_normal_vec_normalized )

                    # Concat Normalize Average Normal Vector
                    if avg_normal_vec.shape[0] == 0 : 
                        avg_normal_vec = avg_normal_vec_normalized
                    else: 
                        avg_normal_vec = torch.cat( [avg_normal_vec, avg_normal_vec_normalized], dim=0 )
                else:                
                    if avg_normal_vec.shape[0] == 0 : avg_normal_vec = torch.tensor([[0,0]], device="cuda")
                    else: avg_normal_vec = torch.cat( [avg_normal_vec, torch.tensor([[0,0]], device="cuda")], dim=0 )

                #wayp_normal_vec.append( normal_vec_list )
            else:
                # 1 Points Occupied
                # Concat Normalize Average Normal Vector
                if avg_normal_vec.shape[0] == 0 : avg_normal_vec = torch.tensor([[0,0]], device="cuda")
                else: avg_normal_vec = torch.cat( [avg_normal_vec, torch.tensor([[0,0]], device="cuda")], dim=0 )
            
            #if skip_num_grad.shape[0] == 0: skip_num_grad = torch.tensor([identical_start_goal], device="cuda").reshape(-1)
            #else: skip_num_grad = torch.cat( [skip_num_grad, torch.tensor([identical_start_goal], device="cuda").reshape(-1)] )
        #print(f'{wayp_normal_vec}')
        #print(f'{avg_normal_vec}')      # [waypoints num, 2]

        output = {
            'avg_normal_vec': avg_normal_vec,       # Num Grad
            'edge_idx'      : edge_idx_list         # Filter Edge IDX
        }

        return output


    def sr_get_start_goal(self, edge_idx):
        idx = torch.arange( edge_idx[0].shape[0] )
        idx_combs = torch.combinations( idx )

        point_A_idx = idx_combs[:,0]
        point_B_idx = idx_combs[:,1]
        
        val = torch.Tensor()
        for combs_iter in range (point_A_idx.shape[0]):
            L1_x = edge_idx[0][ point_A_idx[combs_iter] ] - edge_idx[0][ point_B_idx[combs_iter] ]
            L1_y = edge_idx[1][ point_A_idx[combs_iter] ] - edge_idx[1][ point_B_idx[combs_iter] ]
            L2_dist = torch.sqrt( L1_x**2 + L1_y**2 )

            if val.shape[0] == 0:
                val = L2_dist.reshape(1,-1)
            else:
                val = torch.cat( [ val, L2_dist.reshape(1,-1) ], dim=0 )
        
        #max_idx = torch.argmax(val)
        descending_val, descending_val_idx = torch.sort( val.reshape(1,-1), descending=True )
        descending_val_idx = descending_val_idx.reshape(-1)
        #print(f'start/end idx: {idx_combs[max_idx]}')
        #print(f'start: ({edge_idx[0][idx_combs[max_idx][0]]}, {edge_idx[1][idx_combs[max_idx][0]]})')
        #print(f'end  : ({edge_idx[0][idx_combs[max_idx][1]]}, {edge_idx[1][idx_combs[max_idx][1]]})')
        #start = torch.tensor( [ edge_idx[0][idx_combs[max_idx][0]] , edge_idx[1][idx_combs[max_idx][0]] ] )
        #goal  = torch.tensor( [ edge_idx[0][idx_combs[max_idx][1]] , edge_idx[1][idx_combs[max_idx][1]] ] )
        #start = torch.tensor( [ edge_idx[0][idx_combs[descending_val_idx][0]] , edge_idx[1][idx_combs[descending_val_idx][0]] ] )
        #goal  = torch.tensor( [ edge_idx[0][idx_combs[descending_val_idx][1]] , edge_idx[1][idx_combs[descending_val_idx][1]] ] )

        return descending_val_idx, idx_combs 

    def sr_eval_nvec(self, avg_nvec, edge_idx):     # edge_idx --> (collision wayp, edge_idx_row, edge_idx_col)
        # normal vector of collision waypoint : avg_nvec
        collision_idx = torch.nonzero( self.occupancy ).reshape(-1)
        print(f'average normal vector @ each waypoint: {avg_nvec}')
        print(f'filter edge idx: {edge_idx}, edge idx length {len(edge_idx)}')

        # Get body points on collision waypoints (collision wayp, bodypoints num, 3)
        collision_body_coord = self.sr_body_to_world( self.robot_body, collision_idx )
        collision_body_coord = collision_body_coord.reshape(-1, 3)
        
        # Get Sigma & BP Gradient @ collision waypoints
        autograd, bpgrad, sigma = self.grad_bp( collision_body_coord )

        if torch.isnan( bpgrad ).sum()  or torch.isinf( bpgrad ).sum():
            #print(f'NAN/INF Detection !!!')
            bpgrad = torch.nan_to_num( bpgrad, nan=0, posinf=0, neginf=0 )
        
        # Norm bpgrad
        bpgrad = bpgrad / torch.norm( bpgrad, dim=1 ).reshape(-1,1)

        if torch.isnan( bpgrad ).sum() or torch.isinf( bpgrad ).sum():
            #print(f'NAN/INF Detection !!!')
            bpgrad = torch.nan_to_num( bpgrad, nan=0, posinf=0, neginf=0 )

        bpgrad = bpgrad.reshape( collision_idx.shape[0], self.body_nbins[0], self.body_nbins[1], self.body_nbins[2], -1)
        coord_bpgrad = bpgrad.swapdims( 1, 3 )  # bodypoints bpgrad @ waypoints
        
        edge_bp_cos_sim = torch.Tensor()
        global_bp_cos_sim = torch.Tensor()
        edge_bpgrad_list = torch.Tensor()
        global_bpgrad_list = torch.Tensor()

        if len(edge_idx):
            for p in range (len(edge_idx)):     # Get Collision Waypoints Num
                # Get Edge Gradient
                if len(edge_idx[p]):
                    edge_bpgrad = coord_bpgrad[p,:,edge_idx[p][0],edge_idx[p][1]]
                    edge_bpgrad = edge_bpgrad.reshape(-1,3)
                    avg_edge_bpgrad = torch.mean( edge_bpgrad, 0 )[:2]            

                    inner_p1 = torch.inner(-avg_edge_bpgrad, avg_nvec[p].float())
                    cos1 = inner_p1 / torch.norm(avg_edge_bpgrad) / torch.norm(avg_nvec[p].float())
                    inner_p2 = torch.inner(avg_edge_bpgrad, avg_nvec[p].float())
                    cos2 = inner_p2 / torch.norm(avg_edge_bpgrad) / torch.norm(avg_nvec[p].float())
                    
                    if cos1 > cos2:
                        cos = cos1
                        ebpgrad = -avg_edge_bpgrad
                    else:
                        cos = cos2
                        ebpgrad = avg_edge_bpgrad

                    if edge_bp_cos_sim.shape[0]==0: edge_bp_cos_sim = cos.reshape(-1,1)
                    else: edge_bp_cos_sim = torch.cat( [ edge_bp_cos_sim, cos.reshape(-1,1) ] , dim=0 )
                    
                    if edge_bpgrad_list.shape[0]==0: edge_bpgrad_list = ebpgrad.reshape(1,-1)
                    else: edge_bpgrad_list = torch.cat( [edge_bpgrad_list, ebpgrad.reshape(1,-1)], dim=0 )
                    
                    # Grad Avg @ Entire Body --> entire bpgrad @ body
                    global_bpgrad = torch.mean( coord_bpgrad[p].reshape(-1,3), 1 )[:2]
                    inner_p1 = torch.inner(-global_bpgrad, avg_nvec[p].float())
                    cos1 = inner_p1 / torch.norm(global_bpgrad) / torch.norm(avg_nvec[p].float())
                    inner_p2 = torch.inner(global_bpgrad, avg_nvec[p].float())
                    cos2 = inner_p2 / torch.norm(global_bpgrad) / torch.norm(avg_nvec[p].float())
                    
                    if cos1 > cos2:
                        cos = cos1
                        gbpgrad = -global_bpgrad
                    else:
                        cos = cos2
                        gbpgrad = global_bpgrad

                    if global_bp_cos_sim.shape[0]==0: global_bp_cos_sim = cos.reshape(-1,1)
                    else: global_bp_cos_sim = torch.cat( [ global_bp_cos_sim, cos.reshape(-1,1) ] , dim=0 )

                    if global_bpgrad_list.shape[0]==0: global_bpgrad_list = gbpgrad.reshape(1,-1)
                    else : global_bpgrad_list = torch.cat( [global_bpgrad_list, gbpgrad.reshape(1,-1)] ,dim=0 )

                else:
                    if edge_bp_cos_sim.shape[0]==0: edge_bp_cos_sim = torch.tensor([[0]], device="cuda").reshape(-1,1)
                    else: edge_bp_cos_sim = torch.cat( [ edge_bp_cos_sim, torch.tensor([[0]], device="cuda").reshape(-1,1) ] , dim=0 )
                    
                    if edge_bpgrad_list.shape[0]==0: edge_bpgrad_list = torch.tensor([[0,0]], device="cuda").reshape(1,-1)
                    else: edge_bpgrad_list = torch.cat( [edge_bpgrad_list, torch.tensor([[0,0]], device="cuda").reshape(1,-1)], dim=0 )

                    # Grad Avg @ Entire Body --> entire bpgrad @ body
                    global_bpgrad = torch.mean( coord_bpgrad[p].reshape(-1,3), 1 )[:2]
                    inner_p1 = torch.inner(-global_bpgrad, avg_nvec[p].float())
                    cos1 = inner_p1 / torch.norm(global_bpgrad) / torch.norm(avg_nvec[p].float())
                    inner_p2 = torch.inner(global_bpgrad, avg_nvec[p].float())
                    cos2 = inner_p2 / torch.norm(global_bpgrad) / torch.norm(avg_nvec[p].float())
                    
                    if cos1 > cos2:
                        cos = cos1
                        gbpgrad = -global_bpgrad
                    else:
                        cos = cos2
                        gbpgrad = global_bpgrad

                    if global_bp_cos_sim.shape[0]==0: global_bp_cos_sim = cos.reshape(-1,1)
                    else: global_bp_cos_sim = torch.cat( [ global_bp_cos_sim, cos.reshape(-1,1) ] , dim=0 )

                    if global_bpgrad_list.shape[0]==0: global_bpgrad_list = gbpgrad.reshape(1,-1)
                    else : global_bpgrad_list = torch.cat( [global_bpgrad_list, gbpgrad.reshape(1,-1)] ,dim=0 )

        else:
            # All False Filter Output --> No Edge Grad & Num Grad
            if edge_bp_cos_sim.shape[0]==0: edge_bp_cos_sim = torch.tensor([[0]], device="cuda").reshape(-1,1)
            else: edge_bp_cos_sim = torch.cat( [ edge_bp_cos_sim, torch.tensor([[0]], device="cuda").reshape(-1,1) ] , dim=0 )
            
            if edge_bpgrad_list.shape[0]==0: edge_bpgrad_list = torch.tensor([[0,0]], device="cuda").reshape(1,-1)
            else: edge_bpgrad_list = torch.cat( [edge_bpgrad_list, torch.tensor([[0,0]], device="cuda").reshape(1,-1)], dim=0 )

            # Grad Avg @ Entire Body --> entire bpgrad @ body
            global_bpgrad = torch.mean( coord_bpgrad[0].reshape(-1,3), 1 )[:2]
            inner_p1 = torch.inner(-global_bpgrad, avg_nvec[0].float())
            cos1 = inner_p1 / torch.norm(global_bpgrad) / torch.norm(avg_nvec[0].float())
            inner_p2 = torch.inner(global_bpgrad, avg_nvec[0].float())
            cos2 = inner_p2 / torch.norm(global_bpgrad) / torch.norm(avg_nvec[0].float())
            
            if cos1 > cos2:
                cos = cos1
                gbpgrad = -global_bpgrad
            else:
                cos = cos2
                gbpgrad = global_bpgrad

            if global_bp_cos_sim.shape[0]==0: global_bp_cos_sim = cos.reshape(-1,1)
            else: global_bp_cos_sim = torch.cat( [ global_bp_cos_sim, cos.reshape(-1,1) ] , dim=0 )

            if global_bpgrad_list.shape[0]==0: global_bpgrad_list = gbpgrad.reshape(1,-1)
            else : global_bpgrad_list = torch.cat( [global_bpgrad_list, gbpgrad.reshape(1,-1)] ,dim=0 )


        print(f'collision waypoints         : {collision_idx}')
        print(f'edge bp cos similarity      : {edge_bp_cos_sim.reshape(1,-1)}')
        print(f'global bp cos similarity    : {global_bp_cos_sim.reshape(1,-1)}')
        
        outputs = {
            'edge_cos_sim'      :   edge_bp_cos_sim,
            'global_cos_sim'    :   global_bp_cos_sim,
            'edge_bpgrad'       :   edge_bpgrad_list,
            'global_bpgrad'     :   global_bpgrad_list
        }
        
        return outputs

    def grad_bp(self, x):
        x = x.clone().detach()
        x.requires_grad_(True)
        sigma = self.nerf( x )**2
        sigma.requires_grad_(True)

        autograd_gradient = torch.autograd.grad(sigma, x, grad_outputs=torch.ones_like((sigma), device="cuda"), retain_graph=True, create_graph=True, allow_unused=True)[0]#, allow_unused=True)
        sigma.backward(torch.ones((x.shape[0]), device="cuda"), retain_graph = True)

        return autograd_gradient, x.grad, sigma

################################################################################################################

    def sr_get_cost(self):
        density = self.nerf( self.dh_body_to_world(self.robot_body) )**2        # Input: waypoints, bodypoints, 3(xyz) ==> Output: waypoints, bodypoints
        colision_prob = torch.mean(density, dim = -1)                           # waypoints 
        return colision_prob                                                    # waypoints (avg sigma across body points)
    
    def sr_set_zbound(self, path):
        if path == "stonehenge":
            z_bound = 14
        elif path == "scannetroom":
            z_bound = 16
        elif path == "scannet_472":
            z_bound = 14
        elif path == "scala_room":
            z_bound = 14
        elif path == "replica_FRL_v2":
            z_bound = 15
        elif path == "replica_office3":
            z_bound = 16
        elif path == "replica_apt2":
            z_bound = 15
        elif path == "replica_apt1":
            z_bound = 15
        elif path == "replica_hotel":
            z_bound = 16
        else:
            z_bound = 20
        return z_bound


    def sr_save_tunning_poses( self, filename):
        self.save_poses(self.basefolder /  "replan_poses" / filename)

    

######################################## SR GET WAYP TUNNING VEC #################################################
    def sr_grad_selector(self, num_grad, edge_bpgrad, occupancy_bmap, projected_bmap, edge_cos_sim, plan_mode, edge_skip):
        
        # Get Back-prop Tuning Bit Map
        bp_bmap, num_grad_idx, bp_grad_idx, tunning_idx, numgrad1_bpgrad0 = self.sr_get_bp_tunning_bmap(num_grad, edge_bpgrad, occupancy_bmap, projected_bmap, edge_cos_sim, plan_mode)

        # Get Body Points Grad on Waypoints
        bpgrad = self.sr_get_body_grad()

        # Make BP Grad Zero
        bpgrad[ torch.logical_not( bp_bmap ) ] = 0
        avg_bpgrad = torch.mean( torch.mean( torch.mean( bpgrad, 3 ), 2 ), 1 )
        avg_bpgrad[:,2] = 0
        # Extend BP Grad & Num Grad
        num_grad_extend = torch.cat( [num_grad, torch.zeros((num_grad.shape[0],1),device="cuda" )], dim=1)
        edge_bpgrad_extend = torch.cat( [edge_bpgrad, torch.zeros((edge_bpgrad.shape[0],1),device="cuda" )], dim=1)

        # Replace BP Grad with Num Grad & Edge BPGrad
        for tunning in range ( edge_skip.shape[0] ):
            if edge_skip[tunning] == False:     # Edge
                if plan_mode == "hybrid":
                    # Set Edge Num Vec

                    # High Cos Sim
                    if numgrad1_bpgrad0[tunning]:
                        avg_bpgrad[ tunning_idx[tunning] ] = num_grad_extend[tunning] 

                    # Low Cos Sim
                    else:
                        avg_bpgrad[ tunning_idx[tunning] ] = edge_bpgrad_extend[tunning]
                else:
                    # Set Edge BP Vec
                    avg_bpgrad[ tunning_idx[tunning] ] = edge_bpgrad_extend[tunning]
            else: # No Edge
                avg_bpgrad[ tunning_idx[tunning] ] = -avg_bpgrad[ tunning_idx[tunning] ] / torch.norm(avg_bpgrad[ tunning_idx[tunning] ])

        #if plan_mode == "hybrid":
        #    if num_grad_idx.shape[0] != 0 : 
        #        avg_bpgrad[ num_grad_idx ] = num_grad_extend[torch.logical_not(edge_skip)]
        #    #for skip in range (edge_skip.shape[0]):
        #    #    pdb.set_trace()
        #    #    if not edge_skip[skip]  : avg_bpgrad[ num_grad_idx[skip] ] = num_grad_extend[skip]
        #    #    else                    : avg_bpgrad[ num_grad_idx[skip] ] = -avg_bpgrad[ num_grad_idx[skip] ] / torch.norm(avg_bpgrad[ num_grad_idx[skip] ])

        #if bp_grad_idx.shape[0] != 0 : 
        #    pdb.set_trace()
        #    for skip in range (edge_skip.shape[0]):
        #        if not edge_skip[skip]  : 
        #            if plan_mode == "bp"    : avg_bpgrad[ bp_grad_idx[skip] ] = edge_bpgrad_extend[skip]
        #            else                    : continue  
        #        else                    : avg_bpgrad[ bp_grad_idx[skip] ] = -avg_bpgrad[ bp_grad_idx[skip] ] / torch.norm(avg_bpgrad[ bp_grad_idx[skip] ])

        print(f'final bpgrad: {avg_bpgrad}')


        outputs = {
            'avg_bpgrad'    :   avg_bpgrad
        }

        return outputs

    def sr_sigma_body(self):
        # Body Points on Waypoints
        coords = self.dh_body_to_world( self.robot_body )

        # Sigma @ Body Points
        sigma = self.nerf(coords)**2
        
        # Reshape Sigma to 3D Tensor
        sigma = sigma.reshape( sigma.shape[0], self.body_nbins[0], self.body_nbins[1], self.body_nbins[2] )

        return sigma.swapdims( 1, 3 )

    def sr_get_bp_tunning_bmap(self, num_grad, edge_bpgrad, occupancy_bmap, projected_bmap, edge_cos_sim, plan_mode):
        # Num Grad Waypoints: self.occupancy
        body_sigma = self.sr_sigma_body()
        body_sigma_bmap = body_sigma > self.opt.occupancy_th
        body_occupancy_bmap = torch.sum( torch.sum( torch.sum( body_sigma_bmap, 3 ), 2 ), 1 ) != 0
        body_occupancy_bmap_idx = torch.nonzero( body_occupancy_bmap )

        # Set BP Bmap
        bp_bmap = body_occupancy_bmap
        print(f'occupancy bmap: {bp_bmap}')

        # Num Grad Bmap --> Find Collision Wayp Idx
        high_cosim = edge_cos_sim.reshape(-1) > self.opt.cosim_th     # if true --> select numerical gradient
        ngrad_idx = torch.nonzero( self.occupancy ).reshape(-1)
        num_grad_sel_idx = ngrad_idx[ high_cosim ]
        bp_grad_sel_idx = ngrad_idx[ torch.logical_not(high_cosim) ]

        # BP Grad = False
        if plan_mode == "hybrid":
            bp_bmap[num_grad_sel_idx] = False
            bp_bmap[bp_grad_sel_idx] = True
        elif plan_mode == "bp":
            bp_bmap[num_grad_sel_idx] = True
            bp_bmap[bp_grad_sel_idx] = True
            bp_grad_sel_idx = torch.cat( [bp_grad_sel_idx, num_grad_sel_idx] )
            num_grad_sel_idx = torch.Tensor()
            print(f'bp tunning bmap: {bp_bmap}')

        return bp_bmap, num_grad_sel_idx.reshape(-1), bp_grad_sel_idx.reshape(-1), ngrad_idx, high_cosim

    def sr_get_body_grad(self):
        num_matrices = self.states.size(0)      # waypoints num 
        identity_matrices = torch.eye(3).unsqueeze(0).expand(num_matrices, -1, -1) # expand identity matrix to way points num (batch) 
        coords = self.states[:, :3] # waypoints x,y,z
        coords =  identity_matrices @ self.robot_body.T + coords[..., None]
        coords = coords.swapdims(-1,-2)
        coords = coords.reshape(-1,3)

        autograd, bpgrad, sigma = self.grad_bp( coords )

        bpgrad = bpgrad.reshape( -1, self.body_nbins[0], self.body_nbins[1], self.body_nbins[2], 3)
        coord_bpgrad = bpgrad.swapdims( 1, 3 )        

        return coord_bpgrad

######################################## SR WAYP TUNNING #################################################
    def sr_wayp_tunner(self, tunning_vec, plan_mode):
        self.states.requires_grad_(False)
        if plan_mode == "hybrid" : self.states[:,:3] = self.states[:,:3] + self.opt.tunning_vec_dis * tunning_vec
        elif plan_mode == "bp"   : self.states[:,:3] = self.states[:,:3] + self.opt.bp_tunning_vec_dis * tunning_vec
        self.states.requires_grad_(True)

######################################## SR GET WAYP TUNNING VEC #################################################
    def sr_get_wayp_occupancy(self):
        # Num Grad Waypoints: self.occupancy
        body_sigma = self.sr_sigma_body()
        body_sigma_bmap = body_sigma > self.opt.occupancy_th
        body_occupancy_bmap = torch.sum( torch.sum( torch.sum( body_sigma_bmap, 3 ), 2 ), 1 ) != 0

        return body_occupancy_bmap

######################################## SR GET WAYP TUNNING VEC #################################################
    def sr_get_states(self):
        return self.states
################################################################################################################



    def dh_save_poses_L2_dist(self, filename, mode=None):
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
                # print(f'pos {pos}')
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

            total_L2_distance = 0.0
            for i in range(0, positions.shape[0] - 1):
                L2_distance = torch.dist(positions[i], positions[i+1], p=2)
                total_L2_distance += L2_distance
            print(f'total distance = {total_L2_distance:.4}')

        if mode == "hybrid":
            self.hybrid_dist = total_L2_distance
        else:
            self.bp_dist = total_L2_distance
        
        return total_L2_distance

    def dh_save_poses(self, filename):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        poses = []
        pose_dict = {}


        
        adjacent_groups = []
        current_group = []

        indices = self.indices.tolist()
        for num in indices:
            if not current_group or num == current_group[-1] + 1: 
                current_group.append(num)
            else:  
                adjacent_groups.append(current_group)
                current_group = [num]
        if current_group: 
            adjacent_groups.append(current_group)

        print(adjacent_groups)

        indices_to_extract = []

        for group in adjacent_groups:
            group_start = group[0]
            extracted_indices = [group_start - i for i in range(1,4)]
            indices_to_extract.extend(extracted_indices)

        sorted_indices_to_extract = sorted(indices_to_extract)

        # pose cal 시 2개의 index 오차 보정
        sorted_indices_to_extract = [x + 2 for x in sorted_indices_to_extract]
        print(sorted_indices_to_extract)
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################

        with open(filename,"w+") as f:
            i = 0
            for pos, rot in zip(positions, rot_matrix):
                if i in sorted_indices_to_extract : 
                    # print("h")
                    i += 1
                    # continue
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.cpu().detach().numpy()
                pose[:3, 3]  = pos.cpu().detach().numpy()
                pose[3,3] = 1

                poses.append(pose.tolist())

                i += 1


            
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################


    def sr_get_a_star_states(self):
        self.states = self.bp_states.clone().detach().requires_grad_(True)

    def sr_inc_iters(self, mode):
        if mode == "hybrid": self.hybrid_iters = self.hybrid_iters + 1
        else: self.bp_iters = self.bp_iters + 1

    def sr_get_iteration(self):
        return self.hybrid_iters, self.bp_iters

    def sr_get_distance(self):
        return self.hybrid_dist, self.bp_dist

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
