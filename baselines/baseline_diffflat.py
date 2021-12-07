

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
        print(f"{index=} {t=}")
        print(len(self.polynomials))

        if (index == len(self.polynomials) and index == t):
            return self.polynomials[index - 1].derivative(n).f(t, value) # last point in Piecewise function

        return self.polynomials[index].derivative(n).f(t, value)

    def loss(self):
        return sum( poly.loss() for poly in self.polynomials )

class MinSnap:
    def __init__(self, waypoints, subsample=1):
        self.subsample = subsample

        self.dt = 0.1
        self.mass = 1
        self.J = torch.eye(3)
        self.g = torch.tensor([0,0,-10])

        self.waypoints = waypoints

        segments = waypoints.shape[0] - 1
        self.x = Piecewise(segments, order = 7, loss_derivative = 4)
        self.y = Piecewise(segments, order = 7, loss_derivative = 4)
        self.z = Piecewise(segments, order = 7, loss_derivative = 4)
        self.a = Piecewise(segments, order = 4, loss_derivative = 2)

    def constraints(self):
        constraints = []

        constraints.extend( self.x.constraints() )
        constraints.extend( self.y.constraints() )
        constraints.extend( self.z.constraints() )
        constraints.extend( self.a.constraints() )

        # zero initial velocity
        # constraints.append( self.x.df(1,0) == 0 )
        # constraints.append( self.y.df(1,0) == 0 )
        # constraints.append( self.z.df(1,0) == 0 )
        # constraints.append( self.a.df(1,0) == 0 )

        final_t = self.waypoints.shape[0] - 1
        # constraints.append( self.x.df(1,final_t) == 0 )
        # constraints.append( self.y.df(1,final_t) == 0 )
        # constraints.append( self.z.df(1,final_t) == 0 )
        # constraints.append( self.a.df(1,final_t) == 0 )

        # zero initial, final velocity,acceleration
        for axis in [self.x, self.y, self.z, self.a]:
            for time in [0, final_t]:
                for derivative in [1, 2]:
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
        print( f"{time=}" )
        try:
            out = []
            for t in time:
                T = t/self.dt
                print( f"loop {T=}" )
                out.append([self.x.df(n, T, value=True),
                             self.y.df(n, T, value=True),
                             self.z.df(n, T, value=True),
                             self.a.df(n, T, value=True)])
            return np.squeeze((np.array(out)/ self.dt**n) )
        except TypeError:
            T = time/self.dt
            print( f"noloop {T=}" )
            return np.squeeze(np.array([self.x.df(n, T, value=True),
                         self.y.df(n, T, value=True),
                         self.z.df(n, T, value=True),
                         self.a.df(n, T, value=True)]) / self.dt**n)

    def calc_everything(self):
        finalt = (self.waypoints.shape[0]-1)*self.dt
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
        trust     = torch.norm(needed_acceleration, dim=-1, keepdim=True)

        print(accel.shape)
        print(needed_acceleration.shape)
        print(trust.shape)

        # needs to be pointing in direction of acceleration
        z_axis_body = needed_acceleration/trust
        in_plane_heading = torch.stack( [torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)], dim=-1)

        print(z_axis_body.shape)
        print(in_plane_heading.shape)
        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        omega = torch.zeros( (timesteps, 3) ) # TODO not used for anything since we only need poses
        angular_accel = torch.zeros( (timesteps, 3) )
        actions = torch.zeros( (timesteps, 4) ) 

        return pos, vel, accel, rot_matrix, omega, angular_accel, actions

    def get_full_states(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return torch.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        print(rot_matrix.type())
        print(points.type())
        print(pos.type())
        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def save_data(self, filename):
        positions, vel, _, rot_matrix, omega, _, actions = self.calc_everything()
        # total_cost, colision_loss  = self.get_state_cost()
        total_cost, colision_loss  = torch.zeros((1)), torch.zeros((1))

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


def a_star_init(nerf, start_state, end_state, kernel_size = 5):
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

    return states.clone().detach()




def real():
    cfg = {
            "experiment_name": "stonehenge_L_minsnap_k5",
            "nerf_config_file": 'configs/stonehenge.txt',
            "start_pos": [-0.47, -0.7, 0.1],
            "end_pos": [0.12, 0.51, 0.16],
            "astar": True,
            "astar_kernel": 5,
            "subsample": 1,
            }

    renderer = get_nerf(cfg['nerf_config_file'], need_render=False)

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

    waypoints = a_star_init(renderer.get_density, start_pos, end_pos, kernel_size = kernel)

    traj = MinSnap(waypoints, subsample=cfg['subsample'])
    traj.solve()

    traj.save_data(basefolder / "train" / "0.json")

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


def testing():
    waypoints = np.array( [[0  , -1  ,0,0],
                           [0.5, -0.5,0,0],
                           [1  ,0    ,0,0],
                           [0.5,0.5  ,0,0],
                           [0  ,1    ,0,0]] )

    traj = MinSnap(waypoints)
    traj.solve()


    time = np.linspace(0, 2, 100)

    x = [traj.x.f(t, value = True) for t in time]
    dx = [traj.x.df(1, t, value = True) for t in time]
    y = [traj.y.f(t, value = True) for t in time]
    z = [traj.z.f(t, value = True) for t in time]

    print(time)
    print(x)

    # print(waypoints)

    # print(traj.x.polynomials[0].coef.value)
    # print(traj.x.polynomials[1].coef.value)
    # print(traj.x.polynomials[1].derivative(1).coef.value)

    plt.plot(x, y)
    # plt.plot(time, x)
    # plt.plot(time, dx)
    # plt.plot(time, y)
    # plt.plot(time, z)
    plt.show()



if __name__ == "__main__":
    # testing()
    real()

