import torch
from torch._C import device
import numpy as np
import json

from .math_utils import rot_matrix_to_vec
from .quad_helpers import astar, next_rotation
import pdb
import matplotlib.pyplot as plt

import time
import cv2

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
        
        self.states = states.clone().detach().requires_grad_(True)
        self.initial_accel = torch.tensor([cfg['g'], cfg['g']]).requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(self.body_extent[0, 0], self.body_extent[0, 1], self.body_nbins[0]),
                                            torch.linspace(self.body_extent[1, 0], self.body_extent[1, 1], self.body_nbins[1]),
                                            torch.linspace(self.body_extent[2, 0], self.body_extent[2, 1], self.body_nbins[2])), dim=-1)
        
        self.robot_body = body.reshape(-1, 3)

        if self.CHURCH:
            self.robot_body = self.robot_body/2

        self.epoch = 0
        self.hybrid_sampling_states = None
    def full_to_reduced_state(self, state):
        pos = state[:3]
        R = state[6:15].reshape((3,3))

        x,y,_ = R @ torch.tensor( [1.0, 0, 0 ] )
        angle = torch.atan2(y, x)

        return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()

    
    def dh_init_random(self):
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



                num_points = points

                # 시작점에서 종료점까지의 벡터 계산
                delta = self.end_pos - self.start_pos

                # 시작점과 종료점을 제외한 중간 점 생성
                intermediate_points = [self.start_pos + i * delta / (num_points + 1) for i in range(1, num_points + 1)]

                # 결과를 float 텐서로 변환
                intermediate_points_tensor = np.array(intermediate_points, dtype=np.float32)
                intermediate_points_tensor = torch.tensor(intermediate_points_tensor, dtype=torch.float32)
                #pdb.set_trace()
                states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1)
                #states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1).to(torch.float64)
                randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
                states += randomness

                self.states = states.clone().detach().requires_grad_(True)

                nerf_loss = self.dh_get_cost()
                print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(self.states)}')



                #loop condition check
                wayp_random_count += 1
                wayp_random_flag = (wayp_random_count != self.opt.total_path_num)

                if not wayp_random_flag:
                    print('\033[31m'+f'##--- Total Path Distance : {total_path_dist:.3}, Mean Path Distance : {total_path_dist/(wayp_random_count - wayp_failed_count):.3}---##'+'\033[0m')
                    print(f'Success Paths : {self.opt.total_path_num - ( wayp_failed_count + wayp_collision_count )} ')


        return
    def dh_init(self, start, end, points):
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

        # self.start_pos = np.array(start)
        # self.end_pos = np.array([-0.6, -0.7, 0.2])
        
        num_points = int(points[0])
        
        # 시작점에서 종료점까지의 벡터 계산
        delta = self.end_pos - self.start_pos

        # 시작점과 종료점을 제외한 중간 점 생성
        intermediate_points = [self.start_pos + i * delta / (num_points + 1) for i in range(1, num_points + 1)]

        # 결과를 float 텐서로 변환
        intermediate_points_tensor = np.array(intermediate_points, dtype=np.float32)
        intermediate_points_tensor = torch.tensor(intermediate_points_tensor, dtype=torch.float32)
        #pdb.set_trace()
        states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1)
        #states = torch.cat( [intermediate_points_tensor, torch.zeros( (intermediate_points_tensor.shape[0], 1) ) ], dim=-1).to(torch.float64)
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness

        self.states = states.clone().detach().requires_grad_(True)

        nerf_loss = self.dh_get_cost()

        self.hybrid_sampling_states = states.clone().detach() 
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        self.hybrid_posi = positions.clone().detach()
        print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(self.states)}')
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

    """ 회전 반영없이 1:1 density cost 출력하는 함수
    """
    def dh_get_cost(self):
        # density = self.nerf(self.states[:, :3])
        #pdb.set_trace()
        density = self.nerf( self.dh_body_to_world(self.robot_body) )**2
        # print(density.shape)
        colision_prob = torch.mean(density, dim = -1) 
        return colision_prob # colision_prob*1e6
    
    def dh_body_to_world(self, robot_body):
        num_matrices = self.states.size(0)  
        identity_matrices = torch.eye(3).unsqueeze(0).expand(num_matrices, -1, -1)  
        self_states_tensor = self.states[:, :3]

        # world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################        
        
        pos, rot_matrix_dh = self.dh_calc_everything()

        world_points =  identity_matrices @ robot_body.T + self_states_tensor[..., None]

        return world_points.swapdims(-1,-2) 
    
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
        directions = {} # index에 대한 딕셔너리로 호출용이
        indices = indices.tolist()  # 텐서를 리스트로 해야 i가 숫자로 들어가면서 dict를 숫자로 호출가능
        # directions = []
        for i in indices:
            # 방향, 수직벡터로 이동
            mv_dist = 0.1
            targ_tensor_p1 = self.states[i, :3].clone()
            targ_tensor_p1 += mv_dist * perpendicular_vector # perpvec[i] # perpendicular_vector

            targ_tensor_p1d1 = self.states[i, :3].clone()
            targ_tensor_p1d1 = targ_tensor_p1 + mv_dist * direction_vector #dirvec[i] #direction_vector

            targ_tensor_p2 = self.states[i, :3].clone() 
            targ_tensor_p2 -= mv_dist * perpendicular_vector #perpvec[i] #perpendicular_vector

            targ_tensor_p2d2 = self.states[i, :3].clone() 
            targ_tensor_p2d2 = targ_tensor_p2 + mv_dist * direction_vector # dirvec[i] #direction_vector

            target_states2 = torch.stack([targ_tensor_p1, targ_tensor_p1d1, targ_tensor_p2, targ_tensor_p2d2]) 

            # target_states = torch.tensor(target_states)
            target_loss = self.dh_get_cost_target(target_states2)

            loss_averaged = target_loss.view(-1, 2).mean(dim=1)
            if loss_averaged[0] < loss_averaged[1]: # +-X(-> direction)의 loss가 더 작다면 
                directions[i] = True # 방향을 direction으로
            else: # +-Y의 loss가 더 크다면
                directions[i] = False # 반대로
            print(f'direct target : {target_states2}')
            print(f'inference : {target_loss}')
            print(f'avg : {loss_averaged}')
            # ####################### ##########################################
            # pdb.set_trace()
            # ####################### ##########################################
        print(f'directions: {directions}')
        return directions
    
    def dh_direction_v2(self, indices, direction_vector, perpendicular_vector, dirvec, perpvec):
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
        
        # 인접한 숫자 그룹을 나타내는 리스트 생성
        adjacent_groups = []
        current_group = []

        # 인접한 숫자 그룹 찾기
        for num in sorted(direct.keys()):
            if not current_group or num == current_group[-1] + 1: # 첫 숫자 or 연속된 숫자인 경우
                current_group.append(num)
            else: # 한 그룹이 끝난 경우 저장 및 초기화 
                adjacent_groups.append(current_group)
                current_group = [num]
        # 마지막 그룹 처리
        if current_group: 
            adjacent_groups.append(current_group)

        # 다수값 판단하여 다수의 값으로 바꾸기
        for group in adjacent_groups:
            values = [direct[num] for num in group]
            majority_value = max(set(values), key=values.count) # 1개 유지, 2개 반반은 False로(특수)
            for num in group:
                direct[num] = majority_value
        
        return direct, adjacent_groups
    
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
        #path = astar(occupied, start, end)
        path = astar(occupied, start, end, self.nerf, self.robot_body, side)
        
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
        def interp_tensor(tensor, num_points):
            def linear_interpolation(point1, point2, steps):
                return [point1 + i * (point2 - point1) / (steps - 1) for i in range(steps)]

            result_tensor = []
            for i in range(len(tensor) - 1):
                result_tensor.append(tensor[i])
                result_tensor.extend(linear_interpolation(tensor[i], tensor[i + 1], num_points + 2)[1:-1])
            result_tensor.append(tensor[-1])
            result_tensor = torch.stack(result_tensor)
            return result_tensor
        
        tensor = states[:, :3]
        
        result_tensor = []
        for i in range(0, len(tensor) - 1, 1):
            if torch.any(i == true_indices):
                if i + 2 < len(tensor):
                    result_tensor.append(interp_tensor(tensor[i:i+2], 5))
                elif i == len(tensor) - 2:
                    result_tensor.append(interp_tensor(tensor[-2:], 5))
                elif i == len(tensor) - 1:
                    result_tensor.append(tensor[-1])
            else:
                if i + 2 < len(tensor):
                    result_tensor.append(interp_tensor(tensor[i:i+2], 0))
                elif i == len(tensor) - 2:
                    result_tensor.append(interp_tensor(tensor[-2:], 0))
                elif i == len(tensor) - 1:
                    result_tensor.append(tensor[-1])
            
        result_tensor = torch.cat(result_tensor, dim=0)
        
        states_interp = torch.cat( [result_tensor, torch.zeros( (result_tensor.shape[0], 1) ) ], dim=-1)
        self.states = states_interp.clone().detach().requires_grad_(True)
        self.save_poses(self.basefolder / "init_poses" / (str(f"A_star_hybrid")+".json"))
        self.hybrid_sampling_states = states_interp.clone().detach() 
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        self.hybrid_posi = positions.clone().detach()
        # nerf_loss = self.dh_get_cost()
        # print(f'Hybrid path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(self.states)}')
        
        
    def a_star_random_init(self):
        
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
        occupied_value = maxpool(output[None,None,...])[0,0,...]
        occupied = occupied_value > 0.3

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )
        
        pdb.set_trace()
        ### A* 조절해야하는 대상 => for루프 횟수, start/end 전체 혹은 path지정, occupied z-ax 제한여부, assert off, path smooth off
        for idx in range(100):
            # # Manual [0.28, 0.55, 0.24] -> [-0.36, 0.48, 0.18] == [12.8000, 15.5000, 12.4000] -> [6.4000, 14.8000, 11.8000]
            # start_pos = torch.tensor([13.0, 12.5, 12.0]) # [0.3, 0.25, 0.2]->[-0.7, 0.35, 0.2]
            # end_pos = torch.tensor([3.0, 13.5, 12.0])

            # #Stonehenge 
            # start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, 14)])
            # end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, 14)])

            # #Scannet
            # start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, 16)])
            # end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(0, 20), np.random.uniform(10, 16)])
            # #Scannet 472
            # start_pos = torch.tensor([np.random.uniform(7, 14), np.random.uniform(1, 19), np.random.uniform(10, 14)])
            # end_pos = torch.tensor([np.random.uniform(7, 14), np.random.uniform(1, 19), np.random.uniform(10, 14)])

            # #Replica room2
            # start_pos = torch.tensor([np.random.uniform(2, 18), np.random.uniform(6, 15), np.random.uniform(10, 14)])
            # end_pos = torch.tensor([np.random.uniform(2, 18), np.random.uniform(6, 15), np.random.uniform(10, 14)])
            # #FRL apt 
            # start_pos = torch.tensor([np.random.uniform(5, 15), np.random.uniform(1, 20), np.random.uniform(10, 15)])
            # end_pos = torch.tensor([np.random.uniform(5, 15), np.random.uniform(1, 20), np.random.uniform(10, 15)])
            #Replica office3
            start_pos = torch.tensor([np.random.uniform(1, 19), np.random.uniform(6, 15), np.random.uniform(10, 16)])
            end_pos = torch.tensor([np.random.uniform(1, 19), np.random.uniform(6, 15), np.random.uniform(10, 16)])
            # #Replica apt2     
            # start_pos = torch.tensor([np.random.uniform(3, 19), np.random.uniform(0, 19), np.random.uniform(10, 15)])  #* side / 100
            # end_pos = torch.tensor([np.random.uniform(3, 19), np.random.uniform(0, 19), np.random.uniform(10, 15)])  #* side / 100
            # #Replica apt1    
            # start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(3, 16), np.random.uniform(10, 15)])  #* side / 100
            # end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(3, 16), np.random.uniform(10, 15)])  #* side / 100
            # #Replica hotel 
            # start_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(7, 15), np.random.uniform(10, 16)])  #* side / 100
            # end_pos = torch.tensor([np.random.uniform(0, 20), np.random.uniform(7, 15), np.random.uniform(10, 16)])  #* side / 100
            


            # if idx == 63:
            #     continue 
            if side == 200:
                start = tuple(int(start_pos[i])  for i in range(3) ) 
                end = tuple(int(end_pos[i])  for i in range(3) ) 
                start = tuple(x * 2 for x in start)  #짝수의 경우
                end = tuple(x * 2 for x in end)  #짝수의 경우
            else:
                start = tuple(int(start_pos[i] * side / 100 )  for i in range(3) ) 
                end = tuple(int(end_pos[i] * side / 100)  for i in range(3) ) 

            self.start_pos = np.array(start_pos.cpu())
            self.end_pos = np.array(end_pos.cpu())
            
            # # 첫 랜덤포인트 찍는 것이 직사각 맵 bound라면, 이건 그 중 특정 직사각 구역에 대한 제거
            # x_max = 10 * side / 100
            # y_max = 7 * side / 100
            # if (start[0] < x_max and start[1] < y_max) or (end[0] < x_max and end[1] < y_max): #FRL apt0
            #     print(f"path{idx} occupied")
            #     continue
            
            # if (start[0] < 12 and start[1] > 3) or (end[0] < 12 and end[1] > 3): #rep apt2
            #     print(f"path{idx} occupied")
            #     continue
            # if (start[0] < 8 and start[1] < 9) or (end[0] < 8 and end[1] < 9): #scannet 472
            #     print(f"path{idx} occupied")
            #     continue
            if occupied[start]:
                print(f"path{idx} occupied start {start, end}")
                continue
            elif occupied[end]:
                print(f"path{idx} occupied end {start, end}")
                continue
            
            path = astar(occupied, start, end, self.nerf, self.robot_body, side)
            # path = astar(occupied, start, end, self.nerf, self.robot_body, side, dataset="stonehenge", opt=1)

            if path == None:
                print(f'\033[31m path{idx} S/E colision! \033[0m')
                continue
            elif path == 0:
                print(f'\033[31m path{idx} Failed! \033[0m')
                continue

            #path info
            print('\033[31m \033[43m'+f'path{idx}'+ '\033[0m' + f', pos:{start, end}, float:{start_pos, end_pos}')

            ##---------- 1) occupied idx 확인하기 -------------##
            # save_way = []
            # for i in range(len(path)):
            #     if occupied[path[i]]:  # Check if occupied is True at the current index
            #         save_way.append(i)
            # pdb.set_trace()
            # if save_way: # idx가 있다면 interp
            #     start_index = save_way[0]
            #     end_index = save_way[-1]
            #     print("Start Index:", start_index)
            #     print("End Index:", end_index)
            
            #     ##---------- 2) linear interpolation 수행 -------------##
                
            #     def linear_interpolation_3d(point1, point2, num_points):
            #         interpolated_points = []
            #         for i in range(num_points):
            #             alpha = i / (num_points - 1)
            #             interpolated_point = [
            #                 point1[j] * (1 - alpha) + point2[j] * alpha
            #                 for j in range(3)  # 3D space (x, y, z)
            #             ]
            #             interpolated_points.append(tuple(interpolated_point))
            #         return interpolated_points

            #     point1 = path[start_index]
            #     point2 = path[end_index]
            #     num_points = 15

            #     interpolated_points = linear_interpolation_3d(point1, point2, num_points)
            #     #print(interpolated_points)

            #     path[start_index -1:end_index] = interpolated_points
            
            ##---------- 3) path save -------------##
            squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1
            states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

            self.start_state[None, 0:3] = states[0, :3]
            self.end_state[None, 0:3] = states[-1, :3]

            # ##---------- 4) smooth path (diagram of which states are averaged) -------------## -> A* 에선 필요없음
            # # 1 2 3 4 5 6 7
            # # 1 1 2 3 4 5 6
            # # 2 3 4 5 6 7 7
            # prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
            # next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
            # states = (prev_smooth + next_smooth + states)/3

            self.states = states.clone().detach().requires_grad_(True)

            self.save_poses(self.basefolder / "init_poses" / (str(f"path{idx}")+".json"))

            nerf_loss = self.dh_get_cost()
            print(f'A* path Loss : {nerf_loss}, Occupied : {nerf_loss > 0.3}, waypoints: {len(path)}')
            
        
        pdb.set_trace()
        

        """
        # nerf network, grid size, kernel size, threshold 에서 A*의 grid를 정밀하게 해도 얇은벽의 틈은 반드시 존재, 덜 정밀하게 해도 얇은벽이라. th를 아주낮춰도 높여도 X 
        # 7, 8 idx부근이 문제
        # (8, 16, 14), (8, 15, 14) -> 
        # (8, 16, 14), (8, 15, 14), (7, 15, 14)
        """
        

        
        

        # # convert from index cooredinates
        # squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        # #adding way
        # states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

        # #prevents weird zero derivative issues
        # randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        # states += randomness

        # # smooth path (diagram of which states are averaged)
        # # 1 2 3 4 5 6 7
        # # 1 1 2 3 4 5 6
        # # 2 3 4 5 6 7 7
        # prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
        # next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
        # states = (prev_smooth + next_smooth + states)/3

        # self.states = states.clone().detach().requires_grad_(True)

        """
        path = astar(occupied, start, end)
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness
        self.states = states.clone().detach().requires_grad_(True)
        """
        
        # self.save_poses(self.basefolder / "init_poses" / (str("start")+".json"))
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################

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
        
        for Layer in range(0, 5): # Layer = 4 # z축 기준 0~4층, index사용법: coods_z_proj[Layer][Way] # 0, 500, 5 step
            coods_z_proj.append([ torch.stack([ coods[way][Layer+i, :] for i in range(0, 125, 5) ])               for way in range(0, num_ways) ])
            sigma_z_proj.append([ torch.stack([ sigma[way][Layer+i]    for i in range(0, 125, 5) ]) > threshold   for way in range(0, num_ways) ])
            sigma_z_proj_value.append([ torch.stack([ sigma[way][Layer+i]    for i in range(0, 125, 5) ])         for way in range(0, num_ways) ])
        
        # All Layer z axis projection
        sigma_z_proj_all = [ sigma_z_proj[4][way] | sigma_z_proj[3][way] | sigma_z_proj[2][way] | sigma_z_proj[1][way] | sigma_z_proj[0][way]  for way in range(0, num_ways) ]
        

        # Layer별 occupied bitmap과 value 분포 ratio 출력
        for i in range(num_ways):
            print(f"Way {i} occupied ratio OR: { len( torch.where(sigma_z_proj_all[i] == True)[0] ) / len( sigma_z_proj_all[i] ) * 100 }%")
            occupied_ratio_by_layer = 0
            for L in range(4,-1, -1):
                occupied_ratio_by_layer += len( torch.where(sigma_z_proj[L][i] == True)[0] )
            print(f'Way {i} occupied ratio Sum : {occupied_ratio_by_layer / len( sigma_z_proj_all[i] ) * 20 }%')


        # for L in range(4,-1, -1):
        #     way = self.opt.waypoint
        #     print(f"Way {way} Layer {L} Sigma Values")
        #     for i, value in enumerate(sigma_z_proj_value[L][way], start=1):
        #         print(f'{value:.4e}', end=', ' if i % 10 != 0 else '\n')  
        # for L in range(4,-1, -1):
        #     way = self.opt.waypoint
        #     print(f"Way {way} Layer {L} occupied bitmap | occupied ratio : { len( torch.where(sigma_z_proj[L][way] == True)[0] ) }%")
        #     print(sigma_z_proj[L][way])

        # ideal edge detection : sobel filter -> True idx detecting, Problems : 1) high resolution -> many edges. 2) idx arrange문제 -> 선 형태로 추출안됨  
        sigma_edge_idx = [ torch.where( sigma_z_proj_all[way] )[0] for way in range(0, num_ways) ]
        
        return coods_z_proj, sigma_edge_idx, sigma_z_proj_all
    def test_BP_approx(self, coods, sigma, coods_BP, num_ways, coods_z_proj, sigma_edge_idx):
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
            
            if Layer == 0 or Layer == 1 or Layer == 2 or Layer == 3 or Layer == 4: sigma_edge_idx[4] = torch.tensor([5, 15, 25, 35, 45, 64, 73, 92]) 

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
            global_num_grad[4] = global_num_grad[4].detach().numpy()
            print("#-- Visualize n-vec & midp grad --#")
            # #-- Visualize n-vec & midp grad --#
            # midpoint = midpoint.cpu().detach().numpy()
            # midpoint_BP = midpoint_BP.cpu().detach().numpy()
            # n_vec = n_vec.cpu().detach().numpy()
            # for i in range(len(midpoint)):
            #     midpoint_BP[i] = midpoint_BP[i] / np.linalg.norm(midpoint_BP[i]) * 0.03
            #     n_vec[i] = n_vec[i] / np.linalg.norm(n_vec[i]) * 0.03
            # fig, ax1 = plt.subplots(1, 1, subplot_kw={"projection":"3d"})
            # for i in range(len(midpoint)):
            #     ax1.scatter(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2], c='red', marker='o', alpha=0.5)
            #     ax1.quiver(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2],
            #             midpoint_BP[i, 0], midpoint_BP[i, 1], midpoint_BP[i, 2], color='blue')
            #     ax1.quiver(midpoint[i, 0], midpoint[i, 1], midpoint[i, 2],
            #             n_vec[i, 0], n_vec[i, 1], n_vec[i, 2], color='green')
            #     ax1.quiver(midpoint.mean(0)[0], midpoint.mean(0)[1], midpoint.mean(0)[2],
            #             n_vec.mean(0)[0], n_vec.mean(0)[1], n_vec.mean(0)[2], color='cyan')
            #     ax1.quiver(midpoint.mean(0)[0], midpoint.mean(0)[1], midpoint.mean(0)[2],
            #             global_num_grad[4][0], global_num_grad[4][1], global_num_grad[4][2], color='magenta')
            # ax1.set_xlabel('X')
            # ax1.set_ylabel('Y')
            # ax1.set_title(f"Layer{Layer}, way{way}")
            # plt.show() 는 pdb에서 수동으로 하거나 생략하기
            
            
        return
    def get_edge_nvec(self, coods_z_proj, sigma_z_proj_all, num_ways, sigma_edge_idx):
        n_vec_mean, midpoint_mean = [], []
            
        for way in range(0, num_ways):
            ##--- 필터적용해서 edge idx추출하기 ---##
            arr = sigma_z_proj_all[way].reshape(10, 10).numpy().astype(np.uint8) * 255 # T값을 255(W)로 변환
            
            sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)  # x 방향 Sobel 필터
            sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)  # y 방향 Sobel 필터

            sobel_combined = np.sqrt(sobelx**2 + sobely**2)
            sigma_z_filter = ( torch.tensor(sobel_combined) > 0.3 ).reshape(100)
            
            sigma_edge_idx[way] = torch.where( sigma_z_proj_all[way] & sigma_z_filter.cpu() )[0]
            
            ##--- edge idx에 대해서 nvec 뽑고 평균내서 stack하기 ---##
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
    def dh_learn_init2(self):
        try:
            # 방향벡터 대상변경
            coods, sigma, coods_BP, num_ways = self.get_path_info()
            coods_z_proj, sigma_edge_idx, sigma_z_proj_all = self.get_bitmapInWay(coods, sigma, coods_BP, num_ways)
            # self.test_BP_approx(coods, sigma, coods_BP, num_ways, coods_z_proj, sigma_edge_idx)
            # n_vec_mean = self.get_edge_nvec(coods_z_proj, sigma_z_proj_all, num_ways, sigma_edge_idx)
            
            condition_met = False
            loop = 0
            threshold = 0.3
            
            
            
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
                # print(f'pos {pos}')
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)

            # Path에 대한 L2 거리구하기
            total_L2_distance = 0.0
            for i in range(0, positions.shape[0] - 1):
                L2_distance = torch.dist(positions[i], positions[i+1], p=2)
                total_L2_distance += L2_distance
                # print(f'L2 dist = {L2_distance:.4}')
            print(f'total = {total_L2_distance:.4}')
            # print(f'start: {positions[0]}, end: {positions[-1]}')

    def dh_save_poses(self, filename):
        positions, _, _, rot_matrix, _, _, _ = self.calc_everything()
        poses = []
        pose_dict = {}


        ####### 그룹 앞 3개 점 제거 ######
        
        # 1. 인접한 숫자 그룹을 나타내는 리스트 생성
        adjacent_groups = []
        current_group = []

        indices = self.indices.tolist()
        # 인접한 숫자 그룹 찾기
        for num in indices:
            if not current_group or num == current_group[-1] + 1: # 첫 숫자 or 연속된 숫자인 경우
                current_group.append(num)
            else: # 한 그룹이 끝난 경우 저장 및 초기화 
                adjacent_groups.append(current_group)
                current_group = [num]
        # 마지막 그룹 처리
        if current_group: 
            adjacent_groups.append(current_group)

        print(adjacent_groups)

        # 2. 그룹 앞 3개의 index list 생성
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

            """ 특정행 delete
            
                ####### 그룹 앞 3개 점 제거 ######
            
            # 1. 인접한 숫자 그룹을 나타내는 리스트 생성
            adjacent_groups = []
            current_group = []

            indices = self.indices.tolist()
            # 인접한 숫자 그룹 찾기
            for num in indices:
                if not current_group or num == current_group[-1] + 1: # 첫 숫자 or 연속된 숫자인 경우
                    current_group.append(num)
                else: # 한 그룹이 끝난 경우 저장 및 초기화 
                    adjacent_groups.append(current_group)
                    current_group = [num]
            # 마지막 그룹 처리
            if current_group: 
                adjacent_groups.append(current_group)

            print(adjacent_groups)

            # 2. 그룹 앞 3개의 index list 생성
            indices_to_extract = []

            for group in adjacent_groups:
                group_start = group[0]
                extracted_indices = [group_start - i for i in range(1,4)]
                indices_to_extract.extend(extracted_indices)

            sorted_indices_to_extract = sorted(indices_to_extract)

            print(sorted_indices_to_extract)

            # 삭제할 행을 제외한 새로운 텐서 생성
            poses = torch.stack([row for i, row in enumerate(poses) if i not in sorted_indices_to_extract])

            ####################### ##########################################
            pdb.set_trace()
            ####################### ##########################################
            """
            
            
            pose_dict["poses"] = poses
            json.dump(pose_dict, f, indent=4)
        # ####################### ##########################################
        # pdb.set_trace()
        # ####################### ##########################################

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
