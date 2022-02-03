

import sys
sys.path.append('.')

from load_nerf import get_nerf

import torch

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import json
import shutil
import pathlib

torch.manual_seed(0)
np.random.seed(0)

from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix, next_rotation
from quad_helpers import astar

class Polynomial:
    def __init__(self, coefficients = None, order = 7, start_t = 0, loss_derivative = 4):
        if coefficients is None:
            coefficients = cp.Variable(order)

        # self.coef[i] is coefficient on x**i
        self.coef = coefficients
        self.start_t = start_t
        self.order = order
        self.loss_derivative = loss_derivative

    def derivative(self, number = 1):
        coef = self.coef
        for _ in range(number):
            assert coef
            coef = [n*an for n,an in enumerate(coef)][1:]

        return Polynomial(cp.vstack(coef), self.order, self.start_t)

    def f(self, t, value = False):
        if value:
            return sum(( an.value*(t - self.start_t)**n for n,an in enumerate(self.coef) )).astype(np.float)
        else:
            return sum(( an*(t - self.start_t)**n for n,an in enumerate(self.coef) ))


    def loss(self):
        snap = self.derivative(self.loss_derivative)

        # print( snap.coef)
        size = snap.coef.shape[0]

        # coefficients on polynomial squared, integrated from 0 to 1
        H = np.array([[1/(i+j+1) for i in range(size)] for j in range(size)])

        return cp.quad_form(snap.coef, H)

class Piecewise:
    def __init__(self, number, order = 7, loss_derivative = 4):
        self.order = order
        self.polynomials = [ Polynomial(order = order, start_t=start_t, loss_derivative=loss_derivative) for start_t in range(number)]

    def constraints(self):
        out = []
        for i in range(1, len(self.polynomials)):
            a = self.polynomials[i - 1]
            b = self.polynomials[i]
            out.append( a.f(i) == b.f(i) )
            out.append( a.derivative(1).f(i) == b.derivative(1).f(i) )
            if self.order == 7: #ugly hard hack: for position acceleration should be matched
                out.append( a.derivative(2).f(i) == b.derivative(2).f(i) )
        return out

    def f(self, t, value = False):
        index = int(t)
        if (index == len(self.polynomials) and index == t):
            return self.polynomials[index - 1].f(t, value) # last point in Piecewise function
        return self.polynomials[index].f(t, value)

    def df(self, n, t, value = False):
        index = int(t)
        # print(f"{index=} {t=}")
        # print(len(self.polynomials))

        if (index == len(self.polynomials) and index == t):
            return self.polynomials[index - 1].derivative(n).f(t, value) # last point in Piecewise function

        return self.polynomials[index].derivative(n).f(t, value)

    def loss(self):
        return sum( poly.loss() for poly in self.polynomials )

class MinSnap:
    def __init__(self, waypoints, subsample=1, waypoint_dt = 0.1, nerf = None):
        self.subsample = subsample

        self.dt = waypoint_dt
        self.mass = 1
        self.J = torch.eye(3)
        self.g = torch.tensor([0,0,-10])


        self.waypoints = waypoints

        segments = waypoints.shape[0] - 1
        self.x = Piecewise(segments, order = 7, loss_derivative = 4)
        self.y = Piecewise(segments, order = 7, loss_derivative = 4)
        self.z = Piecewise(segments, order = 7, loss_derivative = 4)
        self.a = Piecewise(segments, order = 4, loss_derivative = 2)

        #used only for cost testing
        self.nerf = nerf
        body = torch.stack( torch.meshgrid( torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.02, 0.02,  5)), dim=-1)
        self.robot_body = body.reshape(-1, 3)

    def constraints(self):
        constraints = []

        constraints.extend( self.x.constraints() )
        constraints.extend( self.y.constraints() )
        constraints.extend( self.z.constraints() )
        constraints.extend( self.a.constraints() )

        final_t = self.waypoints.shape[0] - 1
        # zero initial, final velocity,acceleration
        for axis in [self.x, self.y, self.z, self.a]:
            for time in [0, final_t]:
                for derivative in [1, 2]:
                    # our planner doesn't constrain final trust to be == g
                    # it only constraints the orientation (and omega)
                    if time == final_t and derivative == 2 and axis is self.z:
                        continue
                    constraints.append( axis.df(derivative,time) == 0 )



        for t, waypoint in enumerate(self.waypoints):
            constraints.append( self.x.f(t) == waypoint[0]  )
            constraints.append( self.y.f(t) == waypoint[1]  )
            constraints.append( self.z.f(t) == waypoint[2]  )
            constraints.append( self.a.f(t) == waypoint[3]  )

        return constraints

    def loss(self):
        return self.x.loss() + self.y.loss() + self.z.loss() + self.a.loss()
    
    def solve(self):
        constraints = self.constraints()
        prob = cp.Problem(cp.Minimize( self.loss() ), constraints)
        prob.solve()

    def get_flat_outputs_derivative(self, n, time):
        # print( f"{time=}" )
        try:
            out = []
            for t in time:
                T = t/self.dt
                # print( f"loop {T=}" )
                out.append([self.x.df(n, T, value=True),
                             self.y.df(n, T, value=True),
                             self.z.df(n, T, value=True),
                             self.a.df(n, T, value=True)])
            return np.squeeze((np.array(out)/ self.dt**n) )
        except TypeError:
            T = time/self.dt
            # print( f"noloop {T=}" )
            return np.squeeze(np.array([self.x.df(n, T, value=True),
                         self.y.df(n, T, value=True),
                         self.z.df(n, T, value=True),
                         self.a.df(n, T, value=True)]) / self.dt**n)

    @typechecked
    def calc_everything(self) -> (
            TensorType["states", 3], #pos
            TensorType["states", 3], #vel
            TensorType["states", 3], #accel
            TensorType["states", 3,3], #rot_matrix
            TensorType["states", 3], #omega
            TensorType["states", 3], #angualr_accel
            TensorType["states", 4], #actions
        ):


        finalt = (self.waypoints.shape[0]-1)*self.dt

        dt_effective = self.dt / self.subsample
        time = np.linspace(0, finalt , num=self.subsample * self.waypoints.shape[0], endpoint=False)
        # print(self.waypoints.shape)
        print(time)

        state = self.get_flat_outputs_derivative(0, time) 
        pos = torch.tensor(state[:, :3], dtype=torch.float)
        yaw = torch.tensor(state[:, 3], dtype=torch.float)
        vel = torch.tensor(self.get_flat_outputs_derivative(1, time)[:, :3], dtype=torch.float)
        accel = torch.tensor(self.get_flat_outputs_derivative(2, time)[:, :3], dtype=torch.float)

        timesteps = state.shape[0]

        needed_acceleration = accel - self.g
        thrust     = torch.norm(needed_acceleration, dim=-1, keepdim=True)

        rot_matrix = self.get_rot_matirx(time)

        # math is hard so we compute derivatives numerically
        dh = 1e-3
        rot_matrix_zero = rot_matrix[1:-1, :, :]
        rot_matrix_plus = self.get_rot_matirx(time[1:-1] + dh)
        rot_matrix_minus = self.get_rot_matirx(time[1:-1] - dh)

        print(f"{rot_matrix_zero.shape=}")
        print(f"{rot_matrix_plus.shape=}")
        print(f"{rot_matrix_minus.shape=}")

        omega_plus = rot_matrix_to_vec( rot_matrix_plus @ rot_matrix_zero.swapdims(-1,-2) ) / dh
        omega_minus= rot_matrix_to_vec( rot_matrix_zero @ rot_matrix_minus.swapdims(-1,-2) ) / dh

        #start and end states are constrained to have zero omega
        print(f"{torch.zeros((1,3)).shape=}")
        print(f"{((omega_minus + omega_plus)/2).shape=}")
        omega = torch.cat( [torch.zeros((1,3)),  (omega_minus + omega_plus)/2, torch.zeros((1,3))], dim = 0)
        print(f"{omega.shape=}")

        angular_accel = (omega_plus - omega_minus)/dh
        first_angular_accel = (omega[1,:] - omega[0,:])/dt_effective

        angular_accel = torch.cat( [first_angular_accel[None, :], angular_accel, angular_accel[-1,None,:] ], dim=0)

        torques = (self.J @ angular_accel[...,None])[...,0]
        print(f"{torques.shape=}")
        print(f"{thrust.shape=}")
        actions =  torch.cat([ thrust*self.mass, torques ], dim=-1)

        return pos, vel, accel, rot_matrix, omega, angular_accel, actions

    def get_rot_matirx(self, time):
        state = self.get_flat_outputs_derivative(0, time) 

        yaw = torch.tensor(state[:, 3], dtype=torch.float)
        accel = torch.tensor(self.get_flat_outputs_derivative(2, time)[:, :3], dtype=torch.float)

        needed_acceleration = accel - self.g
        thrust     = torch.norm(needed_acceleration, dim=-1, keepdim=True)
        z_axis_body = needed_acceleration/thrust

        in_plane_heading = torch.stack( [torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)
        return rot_matrix

    def get_full_states(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )


    def get_state_cost(self):
        # copied from quad_planner to offer comparison
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        fz = actions[:, 0]#.to(device)
        torques = torch.norm(actions[:, 1:], dim=-1)#.to(device)

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

        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**4 + colision_prob * 1e6, colision_prob*1e6


    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def save_data(self, filename):
        positions, vel, _, rot_matrix, omega, _, actions = self.calc_everything()
        total_cost, colision_loss  = self.get_state_cost()

        poses = torch.zeros((positions.shape[0], 4,4))
        poses[:, :3, :3] = rot_matrix
        poses[:, :3, 3]  = positions
        poses[:, 3,3] = 1

        full_states = self.get_full_states()

        output = {"colision_loss": colision_loss.detach().numpy().tolist(),
                  "poses": poses.detach().numpy().tolist(),
                  "actions": actions.detach().numpy().tolist(),
                  "total_cost": total_cost.detach().numpy().tolist(),
                  "full_states": full_states.detach().numpy().tolist(),
                  }

        with open(filename,"w+") as f:
            json.dump( output,  f)


def a_star_init(nerf, start_state, end_state, kernel_size = 5, exact_goalpoints = False):
    side = 100 #PARAM grid size

    linspace = torch.linspace(-1,1, side) #PARAM extends of the thing
    # side, side, side, 3
    coods = torch.stack( torch.meshgrid( linspace, linspace, linspace ), dim=-1)
    # kernel_size = 5 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size

    min_value = coods[0,0,0,:]
    side_length = coods[-1,-1,-1,:] - coods[0,0,0,:]
    print(min_value)
    print(side_length)

    output = nerf(coods)
    maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
    #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

    # 20, 20, 20
    occupied = maxpool(output[None,None,...])[0,0,...] > 0.33

    grid_size = side//kernel_size


    #convert to index cooredinates
    start_grid_float = grid_size*(start_state[:3] - min_value)/side_length
    end_grid_float   = grid_size*(end_state  [:3] - min_value)/side_length
    start = tuple(int(start_grid_float[i]) for i in range(3) )
    end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

    print(start, end)
    path = astar(occupied, start, end)
    print(path)

    # convert from index cooredinates
    squares =  side_length * (torch.tensor(path, dtype=torch.float)/grid_size) + min_value
    print(squares)

    #adding yaw
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


    if exact_goalpoints:
        start_with_yaw = torch.cat( [start_state, torch.zeros( (1) ) ], dim=-1)[None,:]
        end_with_yaw = torch.cat( [end_state, torch.zeros( (1) ) ], dim=-1)[None,:]
        states = torch.cat( [start_with_yaw, states, end_with_yaw], dim = 0 )

    return states.clone().detach()



def run_many():
    changes = [{"experiment_name":"random_minsnap_stonehenge_0",
                    "start_pos": [0.55, -0.02, 0.05],
                    "end_pos": [-0.72, 0.17, 0.19],}, 

                {"experiment_name":"random_minsnap_stonehenge_1",
                    "start_pos": [-0.12, -0.92, 0.05],
                    "end_pos": [-0.1, 0.8, 0.19],}, 

                {"experiment_name":"random_minsnap_stonehenge_2",
                    "start_pos": [-0.72, -0.75, 0.1],
                    "end_pos": [0.51, 0.48, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_3",
                    "start_pos": [-0.42, -0.75, 0.1],
                    "end_pos": [0.21, 0.48, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_4",
                    "start_pos": [-0.12, -0.75, 0.1],
                    "end_pos": [-0.09, 0.48, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_5",
                    "start_pos": [0.18, -0.75, 0.1],
                    "end_pos": [-0.39, 0.48, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_6",
                    "start_pos": [0.48, -0.75, 0.1],
                    "end_pos": [-0.69, 0.48, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_7",
                    "start_pos": [0.48, -0.42, 0.1],
                    "end_pos": [-0.71, 0.83, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_8",
                    "start_pos": [-0.72, -0.12, 0.1],
                    "end_pos": [0.49, 0.83, 0.16],}, 

                {"experiment_name":"random_minsnap_stonehenge_9",
                    "start_pos": [-0.72, -0.42, 0.1],
                    "end_pos": [0.49, 0.23, 0.16],}, ]

    for change in changes:
        cfg = { "experiment_name": None,
                "nerf_config_file": 'configs/stonehenge.txt',
                "start_pos": None,
                "end_pos": None,
                "astar": True,
                "astar_kernel": 5,
                "minsnap_subsample": 1,
                }

        print(change)
        cfg.update(change)
        print(cfg)
        run_planner(cfg)

def real():
    cfg = { "experiment_name": "minsnap_stonehenge_compare1",
            "nerf_config_file": 'configs/stonehenge.txt',
            "start_pos": [-0.47, -0.7, 0.1],
            "end_pos": [0.12, 0.51, 0.16],
            "astar": True,
            "astar_kernel": 5,
            "minsnap_subsample": 1,
            }

    run_planner(cfg)

def run_planner(cfg):

    experiment_name = cfg['experiment_name']
    renderer = get_nerf(cfg['nerf_config_file'], need_render=False)
    start_pos = torch.tensor(cfg['start_pos'])
    end_pos = torch.tensor(cfg['end_pos'])
    assert cfg['astar']
    kernel = cfg['astar_kernel']

    basefolder = "experiments" / pathlib.Path(experiment_name)
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "train").mkdir()

    print("created", basefolder)
    (basefolder / 'cfg.json').write_text(json.dumps(cfg))

    waypoints = a_star_init(renderer.get_density, start_pos, end_pos, kernel_size = kernel, exact_goalpoints = True)

    traj = MinSnap(waypoints, subsample=cfg['minsnap_subsample'], nerf=renderer.get_density)
    traj.solve()

    traj.save_data(basefolder / "train" / "0.json")

    return

    quadplot = QuadPlot()
    quadplot.trajectory( traj, "g" )

    ax = quadplot.ax_graph

    pos, vel, accel, _, omega, _, actions = traj.calc_everything()
    actions = actions.cpu().detach().numpy()
    pos = pos.cpu().detach().numpy()
    vel = vel.cpu().detach().numpy()
    omega = omega.cpu().detach().numpy()

    ax.plot(pos[...,0], label="px")
    ax.plot(vel[...,0], label="vx")
    ax.plot(accel[...,0], label="ax")
    ax.plot(accel[...,1], label="ay")
    ax.plot(accel[...,2], label="az")
    # ax.plot(actions[...,1], label="tx")
    # ax.plot(actions[...,2], label="ty")
    # ax.plot(actions[...,3], label="tz")
    ax.legend()

    quadplot.show()


def compare_loss():
    # f = 'experiments/ours_stonehenge_compare1/train/139.json'
    f = 'experiments/ours_stonehenge_compare1/train/1.json'
    data = json.load(open(f))

    renderer = get_nerf('configs/stonehenge.txt', need_render=False)

    poses = np.array(data['poses'])
    positions = poses[:, :3 , 3]
    waypoints = np.concatenate([positions, np.zeros((poses.shape[0], 1))], axis=-1)
    print(waypoints.shape)
    traj = MinSnap(waypoints, subsample=1, nerf=renderer.get_density)
    traj.solve()

    pos, vel, accel, _, omega, _, actions = traj.calc_everything()
    total_cost, col_cost = traj.get_state_cost()
    ctrl_cost = total_cost - col_cost

    ours_actions = np.array( data['actions'])
    ours_omega = np.array( data['full_states'])[:, -3:]
    ours_total_cost = np.array(data['total_cost'])
    ours_col_cost = np.array(data['colision_loss'])

    ours_ctrl_cost = ours_total_cost - ours_col_cost

    print(total_cost)
    print(ours_total_cost)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    ax1.plot(actions[...,0], label="df_t0")
    ax1.plot(ours_actions[...,0], label="our_t0")
    ax1.legend()

    ax2.plot(actions[...,1], label="df_tx")
    ax2.plot(ours_actions[...,1], label="our_tx")
    ax2.legend()

    ax3.plot(actions[...,2], label="df_ty")
    ax3.plot(ours_actions[...,2], label="our_ty")
    ax3.legend()

    ax4.plot(actions[...,3], label="df_tz")
    ax4.plot(ours_actions[...,3], label="our_tz")
    ax4.legend()

    ax5.plot(ctrl_cost, label="df_ctrl_cost")
    ax5.plot(ours_ctrl_cost, label="our_ctrl_cost")
    ax5.legend()

    ax6.plot(col_cost, label="df_col_cost")
    ax6.plot(ours_col_cost, label="our_col_cost")
    ax6.legend()

    plt.show()


def testing():
    waypoints = np.array( [[0  , -1  ,0,0],
                           [0.5, -0.5,0,0],
                           [1  ,0    ,0,0],
                           [0.5,0.5  ,0,0],
                           # [0.8,0.8  ,0,0],
                           [0  ,1    ,0,0]] )

    traj = MinSnap(waypoints, subsample=10)
    traj.solve()

    quadplot = QuadPlot()
    quadplot.trajectory( traj, "g" )

    ax = quadplot.ax_graph

    pos, vel, accel, _, omega, _, actions = traj.calc_everything()
    actions = actions.cpu().detach().numpy()
    pos = pos.cpu().detach().numpy()
    vel = vel.cpu().detach().numpy()
    omega = omega.cpu().detach().numpy()

#     ax.plot(pos[...,0], label="px")
#     ax.plot(vel[...,0], label="vx")
#     ax.plot(accel[...,0], label="ax")
#     ax.plot(accel[...,1], label="ay")
#     ax.plot(accel[...,2], label="az")
#     ax.plot(actions[...,0], label="t0")
    ax.plot(omega[...,0], label="wx")
    ax.plot(omega[...,1], label="wy")
    ax.plot(omega[...,2], label="wz")
    ax.plot(actions[...,1], label="tx")
    ax.plot(actions[...,2], label="ty")
    ax.plot(actions[...,3], label="tz")
    ax.legend()

    quadplot.show()


if __name__ == "__main__":
    # testing()
    # real()
    compare_loss()
    # run_many()

