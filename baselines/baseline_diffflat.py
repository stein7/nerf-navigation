

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
            return sum(( an.value*(t - self.start_t)**n for n,an in enumerate(self.coef) ))
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

        if (index == len(self.polynomials) and index == t):
            return self.polynomials[index - 1].derivative(n).f(t, value) # last point in Piecewise function

        return self.polynomials[index].derivative(n).f(t, value)

    def loss(self):
        return sum( poly.loss() for poly in self.polynomials )

class Trajectory:
    def __init__(self, waypoints):

        self.dt = 0.1
        self.mass = 1
        self.J = np.eye(3)
        self.g = np.array([0,0,-10])

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
        constraints.append( self.x.df(1,0) == 0 )
        constraints.append( self.y.df(1,0) == 0 )
        constraints.append( self.z.df(1,0) == 0 )
        constraints.append( self.a.df(1,0) == 0 )

        final_t = self.waypoints.shape[0] - 1
        constraints.append( self.x.df(1,final_t) == 0 )
        constraints.append( self.y.df(1,final_t) == 0 )
        constraints.append( self.z.df(1,final_t) == 0 )
        constraints.append( self.a.df(1,final_t) == 0 )

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
                out.append([self.x.df(n, T), self.y.df(n, T), self.z.df(n, T), self.a.df(n, T)]) / self.dt**n
            return np.array(out)
        except TypeError:
            T = time/self.dt
            print( f"noloop {T=}" )
            return np.array([self.x.df(n, T), self.y.df(n, T), self.z.df(n, T), self.a.df(n, T)]) / self.dt**n

    def calc_everything(self):
        time = np.arange(self.waypoints.shape[0]) * self.dt

        pos = self.get_flat_outputs_derivative(0, time)
        vel = self.get_flat_outputs_derivative(1, time)
        accel = self.get_flat_outputs_derivative(2, time)

        needed_acceleration = accel + self.g
        trust     = torch.norm(needed_acceleration, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = current_accel/trust
        in_plane_heading = torch.stack( [torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        omega = np.zeros( (timesteps, 3) ) # TODO not used for anything since we only need poses
        angular_accel = np.zeros( (timesteps, 3) )
        actions = np.zeros( (timesteps, 4) ) 

        return pos, vel, accel, rot_matrix, omega, angular_accel, actions

    def get_full_states(self):
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()
        return np.cat( [pos, vel, rot_matrix.reshape(-1, 9), omega], dim=-1 )

    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)


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

    renderer = get_nerf('configs/playground.txt', need_render=False)
    experiment_name = "playground_slide_diffflat"
    start_pos = torch.tensor([-0.3, -0.27, 0.06])
    end_pos = torch.tensor([0.02, 0.58, 0.65])
    waypoints = a_star_init(renderer.get_density, start_pos, end_pos)

    traj = Trajectory(waypoints)
    traj.solve()


    quadplot = QuadPlot()
    quadplot.trajectory( traj, "g" )
    quadplot.show()


def testing():
    waypoints = np.array( [[0,-1,0,0], [1,0,0,0], [0,1,0,0]] )
    #TODO verify dimentionality

    traj = Trajectory(waypoints)
    traj.solve()


    time = np.linspace(0, 2, 100)

    x = [traj.x.f(t, value = True) for t in time]
    dx = [traj.x.df(1, t, value = True) for t in time]
    y = [traj.y.f(t, value = True) for t in time]
    z = [traj.z.f(t, value = True) for t in time]

    print(time)
    print(x)

    # print(traj.x.polynomials[0].coef.value)
    # print(traj.x.polynomials[1].coef.value)
    # print(traj.x.polynomials[1].derivative(1).coef.value)

    plt.plot(x, y)
    # plt.plot(time, x)
    # plt.plot(time, dx)
    # plt.plot(time, y)
    # plt.plot(time, z)
    plt.show()


# prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
#                  [G @ x <= h,
#                   A @ x == b])
# prob.solve()


if __name__ == "__main__":
    # testing()
    real()

