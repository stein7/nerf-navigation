import torch
from torch._C import device
import numpy as np
import json

from .math_utils import rot_matrix_to_vec
from .quad_helpers import astar, next_rotation
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Planner:
    def __init__(self, start_state, end_state, cfg, density_fn):
        self.nerf = density_fn

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

    def full_to_reduced_state(self, state):
        pos = state[:3]
        R = state[6:15].reshape((3,3))

        x,y,_ = R @ torch.tensor( [1.0, 0, 0 ] )
        angle = torch.atan2(y, x)

        return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()

    def a_star_init(self):
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

        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        print(start, end)
        path = astar(occupied, start, end)

        self.path = path
        """ 1) occupied 확인하기
        """ 
        # paths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # occupied = [False, True, True, True, False, False, False, False, False, False, False]
        save_way = []

        for i in range(len(path)):
            if occupied[path[i]]:  # Check if occupied is True at the current index
                save_way.append(i)

        if save_way:
            start_index = save_way[0]
            end_index = save_way[-1]
            print("Start Index:", start_index)
            print("End Index:", end_index)
        else:
            print("No occupied points found in the path.")

        """ 2) linear interpolation 수행
        """
        def linear_interpolation_3d(point1, point2, num_points):
            interpolated_points = []
            for i in range(num_points):
                alpha = i / (num_points - 1)
                interpolated_point = [
                    point1[j] * (1 - alpha) + point2[j] * alpha
                    for j in range(3)  # 3D space (x, y, z)
                ]
                interpolated_points.append(tuple(interpolated_point))
            return interpolated_points

        point1 = path[start_index]
        point2 = path[end_index]
        num_points = 10

        interpolated_points = linear_interpolation_3d(point1, point2, num_points)
        print(interpolated_points)

        path[2:4] = interpolated_points

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

        """
        path = astar(occupied, start, end)
        squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

        states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)
        randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
        states += randomness
        self.states = states.clone().detach().requires_grad_(True)
        """
        
        

        ####################### ##########################################
        pdb.set_trace()
        ####################### ##########################################

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

    def dh_direction(self, indices, direction_vector, perpendicular_vector):
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
            targ_tensor_p1 += 0.2 * perpendicular_vector

            targ_tensor_p1d1 = self.states[i, :3].clone()
            targ_tensor_p1d1 = targ_tensor_p1 + 0.2 * direction_vector

            targ_tensor_p2 = self.states[i, :3].clone() 
            targ_tensor_p2 -= 0.2 * perpendicular_vector

            targ_tensor_p2d2 = self.states[i, :3].clone() 
            targ_tensor_p2d2 = targ_tensor_p2 + 0.2 * direction_vector

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
    
    


    def dh_learn_init(self):
        # opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)
        # opt = torch.optim.SGD(self.params(), lr=self.lr)
        try:
            condition_met = False
            
            loop = 0
            # 임계값 설정
            threshold = 1e+0
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
            
            # 방향 벡터 계산
            direction_vector = self.end_pos - self.start_pos
            # direction_vector /= np.linalg.norm(direction_vector)
            direction_vector = torch.tensor(direction_vector, device='cuda', dtype=torch.float32)

            # 수직인 방향 벡터 생성
            perpendicular_vector = torch.tensor([direction_vector[1], -direction_vector[0], 0.0], device='cuda', dtype=torch.float32)
            # perpendicular_vector /= torch.norm(perpendicular_vector)

                ####### direction 판단 ######## 
            
            # directions = {} # index에 대한 딕셔너리로 호출용이
            directions = {}
            directions = self.dh_direction(indices, direction_vector, perpendicular_vector)
            
            # Grouping & 다수의 판단따르기 (특수 : 2개의 경우 False로)
            directions, adjacent_groups = self.dh_direction_tuning(directions)
            
            # ####################### ##########################################
            # pdb.set_trace()
            # ####################### ##########################################
            
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

                        ####### grad 방향 제어(xy평면 일반화 상태) ######## 
                    self.states.grad[indices, 2] = 0 # z축 계산결과는 0으로 초기화
                    # perpendicular_vector와의 내적 계산하여 대입
                    dot_product = torch.sum(self.states.grad[indices, :3] * perpendicular_vector, dim=1)
                    self.states.grad[indices, :3] = dot_product.view(-1, 1) * perpendicular_vector.view(1, -1)
                    
                    # print(f"per:{perpendicular_vector}")
                    # print(f"previous:{self.states.grad[indices, :3]}")
                    # indices = indices.tolist()
                    # directions_list = [directions.get(i, False) for i in range(8)]
                    """ 방향
                    """ 
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

                    

                    # for i in range(len(indices)):
                    #     dot_product_ins = dot_product[i]
                        
                    #     if dot_product_ins < 0: # > : perp 동일, < : perp 반대
                    #         self.states.grad[indices[i], :3] *= -1
                    # print(f"new:{self.states.grad[indices, :3]}")

                    # negative_indices = self.states.grad[:, 0] < 0 # 왼쪽으로만 가도록 grad 조절
                    # self.states.grad[negative_indices, 0] *= -1


                    # 해당 waypoint 추적하기
                    # print(f"tracing point : {loop}loop, {it}iter, {nerf_loss[3]}loss")
                    # print(f"grad : {self.states.grad[3, :3]}, way : {self.states[3, :3]}")
                    # print(self.states.grad[:,:3])
                    """ 연속된 waypoint에 대해서는 같은 방향으로 튜닝이되도록 grad를 조절해야함
                    # self.states.grad[indices] = -self.states.grad[indices] # 이방식은 제어어려워서 폐기
                    # 방법1. 방향 시나리오를 먼저 정해서 if로 부호를 제한
                    for i in indices:
                        if self.states.grad[indices, 0] < 0:
                            self.states.grad[indices, 0] *= -1
                    # 방법2. 인접한 waypoint를 주시대상으로, 먼저 끝난 놈에게 맞게 if로 부호를 동일하게 -> 먼저끝난놈의 순서예측 어려움, 부호를 추적(self.state 해당행의 변화를 관찰)
                    """
                    opt.step()
                    
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

                    # save_step = 50
                    # if it%save_step == 0:
                    #     if hasattr(self, "basefolder"):
                    #         self.save_poses(self.basefolder / "init_poses" / (str(it//save_step)+".json"))
                    #         self.save_costs(self.basefolder / "init_costs" / (str(it//save_step)+".json"))
                    #     else:
                    #         print("Warning: data not saved!")


                    # ####################### ##########################################
                    # pdb.set_trace()
                    # ####################### ##########################################
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