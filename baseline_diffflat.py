

from load_nerf import get_nerf

import cvxpy as cp

torch.manual_seed(0)
np.random.seed(0)

from quad_helpers import Simulator, QuadPlot
from quad_helpers import rot_matrix_to_vec, vec_to_rot_matrix, next_rotation
from quad_helpers import astar



class Polynomial:
    def __init__(self, coefficients = None, order = 5):
        if coefficients is None:
            self.coef = cp.Variable(order, 3)

        self.coef = coefficients
        # self.coef[i] is coefficient on x**i

    def derivative(self, number = 1):
        coef = self.coef
        for _ in range(number):
            assert coef
            coef = [n*an for n,an in enumerate(coef)][1:, :]

        return Polynomial(coef)

    def f(self, t):
        return sum( ( an * t**n for n,an in enumerate(coef) ) )

    def loss(self):

        snap = self.derivative(3)
        lambada_xyz = 1
        lambada_theta = 1
        scaled = lambada_xyz * snap.coef[:, :3] + lambada_theta * snap.coef[:, 3:]


        cp.sum(scaled[:, None, :] * scaled[None, :, :])


    # def drone_direction(self):
    #     thrust = self.derivative(2)
    #     thrust.coef[0] -= np.array([0,0,-9.8])

class Piecewise:
    def __init__(self, number):
        self.polynomials = [ Polynomial() for _ in range(number)]

    def constraints(self):
        out = []
        for i in range(1, len(self.polynomials)):
            a = self.polynomials[i - 1]
            b = self.polynomials[i]
            out.append( a.f(i) == b.f(i) )
            out.append( a.derivative(1).f(i) == b.derivative(1).f(i) )
            out.append( a.derivative(2).f(i) == b.derivative(2).f(i) )
        return out

    def f(self, t):
        index = int(t)
        return self.polynomials[index].f(t)

    # def loss(self):
    #     return sum( self.polynomials.loss() )

def testing():
    waypoints = np.array( [[0,-1,0,0], [1,0,0,0], [0,1,0,0]] )
    #TODO verify dimentionality


    traj = Piecewise( len(waypoints) - 1 ) 

    constraints = traj.constraints()

    # zero initial velocity
    constraints.append( traj.polynomials[0].derivative(1) == np.array([0,0,0,0]) )
    for t, waypoint in enumerate(waypoints):
        constraints.append( [traj.f(t) == waypoint] )




# prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
#                  [G @ x <= h,
#                   A @ x == b])
# prob.solve()


if __name__ == "__main__":
    testing()

#def main():

#    church = False
#    astar = True
#    kernel = 5

#    start_R = vec_to_rot_matrix( torch.tensor([0.0,0.0,0]))
#    end_R = vec_to_rot_matrix( torch.tensor([0.0,0.0,0]))

#    cfg = {"T_final": 2,
#            "steps": 20,
#            "lr": 0.01,
#            "epochs_init": 2500,
#            "fade_out_epoch": 0,
#            "fade_out_sharpness": 10,
#            "epochs_update": 250,
#            }

#    #stonehenge
#    renderer = get_nerf('configs/stonehenge.txt')
#    experiment_name = "stonehenge_with_fan_line" 
#    start_pos = torch.tensor([0.39, -0.67, 0.2])
#    end_pos = torch.tensor([-0.4, 0.55, 0.16])
#    astar = False
#    kernel = 4




#    start_state = torch.cat( [start_pos, torch.tensor([0,0,0]), start_R.reshape(-1), torch.zeros(3)], dim=0 )
#    end_state   = torch.cat( [end_pos,   torch.zeros(3), end_R.reshape(-1), torch.zeros(3)], dim=0 )

#    LOAD = False

#    basefolder = "experiments" / pathlib.Path(experiment_name)

#    if not LOAD:
#        if basefolder.exists():
#            print(basefolder, "already exists!")
#            if input("Clear it before continuing? [y/N]:").lower() == "y":
#                shutil.rmtree(basefolder)
#        basefolder.mkdir()
#        (basefolder / "train").mkdir()

#    print("created", basefolder)

#    traj = System(renderer, start_state, end_state, cfg)
#    traj.a_star_init()

#    flatoutputs = traj.states




#    save = Simulator(start_state)
#    save.copy_states(traj.get_full_states())

#    quadplot = QuadPlot()
#    traj.plot(quadplot)
#    quadplot.trajectory( sim, "r" )
#    quadplot.trajectory( save, "b", show_cloud=False )
#    quadplot.show()

