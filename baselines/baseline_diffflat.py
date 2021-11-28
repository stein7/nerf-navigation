

import sys
sys.path.append('.')

from load_nerf import get_nerf

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(0)
np.random.seed(0)

from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix, next_rotation
from quad_helpers import astar

class Polynomial:
    def __init__(self, coefficients = None, order = 7, start_t = 0):
        if coefficients is None:
            coefficients = cp.Variable(order)

        # self.coef[i] is coefficient on x**i
        self.coef = coefficients
        self.start_t = start_t
        self.order = order

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


    def loss(self, derivative_number = 4):
        snap = self.derivative(derivative_number)

        # print( snap.coef)
        size = snap.coef.shape[0]
        assert size == 3

        # coefficients on polynomial squared, integrated from 0 to 1
        H = np.array([[1/(i+j+1) for i in range(size)] for j in range(size)])

        return cp.quad_form(snap.coef, H)

    # def drone_direction(self):
    #     thrust = self.derivative(2)
    #     thrust.coef[0] -= np.array([0,0,-9.8])

class Piecewise:
    def __init__(self, number, order = 7):
        self.polynomials = [ Polynomial(order = order, start_t=start_t) for start_t in range(number)]

    def constraints(self):
        out = []
        for i in range(1, len(self.polynomials)):
            a = self.polynomials[i - 1]
            b = self.polynomials[i]
            out.append( a.f(i) == b.f(i) )
            out.append( a.derivative(1).f(i) == b.derivative(1).f(i) )
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

        self.waypoints = waypoints

        segments = waypoints.shape[0] - 1
        self.x = Piecewise(segments)
        self.y = Piecewise(segments)
        self.z = Piecewise(segments)
        # self.yaw = Piecewise(

    def f(self, t):
        return [self.x.f(t), self.y.f(t), self.z.f(t)]

    def df(self, n, t):
        [self.x.df(m, t), self.y.df(m, t), self.z.df(m, t)]


    def constraints(self):
        constraints = []

        constraints.extend( self.x.constraints() )
        constraints.extend( self.y.constraints() )
        constraints.extend( self.z.constraints() )

        # zero initial velocity
        # constraints.append( traj.polynomials[0].derivative(1) == np.array([0,0,0,0]) )

        constraints.append( self.x.df(1,0) == 0 )
        constraints.append( self.y.df(1,0) == 0 )
        constraints.append( self.z.df(1,0) == 0 )

        final_t = self.waypoints.shape[0] - 1
        constraints.append( self.x.df(1,final_t) == 0 )
        constraints.append( self.y.df(1,final_t) == 0 )
        constraints.append( self.z.df(1,final_t) == 0 )

        for t, waypoint in enumerate(self.waypoints):
            constraints.append( self.x.f(t) == waypoint[0]  )
            constraints.append( self.y.f(t) == waypoint[1]  )
            constraints.append( self.z.f(t) == waypoint[2]  )

        return constraints

    def loss(self):
        return self.x.loss() + self.y.loss() + self.z.loss()

    
    def solve(self):
        constraints = self.constraints()
        prob = cp.Problem(cp.Minimize( self.loss() ), constraints)
        prob.solve()


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
    testing()

