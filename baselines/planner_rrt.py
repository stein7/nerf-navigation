
import sys
sys.path.append('.')
# run from toplevel directory or imports will fail

import numpy as np
import torch

import json
import shutil
import pathlib

import trimesh
from skimage import measure

from load_nerf import get_nerf

from baselines.baseline_diffflat import MinSnap

torch.manual_seed(0)
np.random.seed(0)


class RRT:
    def __init__(self, start_state, end_state, mesh):
        self.start_state = start_state
        self.end_state = end_state
        self.mesh = mesh

        # map from child to parent point
        self.graph = { self.hashable(start_state) : None }
    
        while True:
            # pick random point (pick goal point with prob)
            use_final_goal = np.random.random() < 0.05
            if use_final_goal:
                new_point = self.end_state
                print("using final", new_point)
            else:
                new_point = 2 * np.random.random((3)) - 1
                print("using random", new_point)

            if self.in_colision(new_point):
                continue
                print("in collision")

            # find closest point in tree
            closest = self.find_closest(new_point)
            print("found closest", closest)

            successful = self.subdivide_line(new_point, closest)

            if successful and use_final_goal:
                break
            # generate stright path to new point
            # for each new point check colision
            # add to tree

        # walk the tree from the goal
        new_point = self.hashable(new_point)
        reverse_waypoints = []
        while True:
            reverse_waypoints.append(new_point)
            new_point = self.graph[new_point]
            if new_point == None:
                break

        print(reverse_waypoints)
        self.waypoints = np.array( list(reversed(reverse_waypoints) ))
        print(self.waypoints.shape)

        self.waypoints = np.hstack( [self.waypoints, np.zeros(( self.waypoints.shape[0], 1))])




    def hashable(self, point):
        return tuple( point.round(5) )


    def subdivide_line(self, target, tree_node):
        normal = target - tree_node
        distance = np.linalg.norm(normal)
        normal = normal/distance

        step_size = 0.15
        prev_point = self.hashable(tree_node)

        # goes from the tree_node (exclusive) to the target (inclusive)
        for point in reversed( np.linspace(target, tree_node, int(distance//step_size), endpoint=False) ):
            print("checking", point)
            if self.in_colision(point):
                return False
                print("colised")
            tuple_point = self.hashable(point)
            self.graph[tuple_point] = prev_point
            prev_point = tuple_point

        print("successgul")
        return True

    def find_closest(self, point):
        # horribly slow etc
        min_distance = float("inf")
        min_point = None
        for node in self.graph.keys():
            distance = np.sum( (point - np.array(node))**2 )
            if distance < min_distance:
                min_distance = distance
                min_point = node

        return np.array(min_point)

    def in_colision(self, point):
        distances = trimesh.proximity.ProximityQuery(self.mesh).signed_distance(point.reshape(1,3))

        # how big is the robot # negative is outside
        # if distances[0] < -0.05:
        if distances[0] < -0.06:
            return False

        return True



    
    def get_distance_to_go(self, state):
        pass

def get_mesh(renderer):
    nerf = renderer.get_density

    side = 40
    linspace = torch.linspace(-1,1, side)
    spacing = (linspace[-1] - linspace[0])/(len(linspace)-1)

    linspacez = torch.linspace(-0.1,0.9, side//2)
    spacingz = (linspacez[-1] - linspacez[0])/(len(linspacez)-1)

    # side, side, side, 3
    coods = torch.stack( torch.meshgrid( linspace, linspace, linspacez ), dim=-1)

    # side, side, side
    voxels = -nerf(coods) # makres normals face correctly
    # print(voxels)

    voxels[0,:,:] = 0
    voxels[:,0,:] = 0
    voxels[:,:,0] = 0

    voxels[-1,:,:] = 0
    voxels[:,-1,:] = 0
    voxels[:,:,-1] = 0

    vertices, faces, normals, _ = measure.marching_cubes(voxels.detach().numpy(), spacing=(spacing, spacing, spacingz))
    vertices += np.array([ linspace[0] - spacing/2, linspace[0] - spacing/2, linspacez[0] - spacingz/2] )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return mesh

def testing():
    cfg = {
            "experiment_name": "rrt_test",
            "nerf_config_file": 'configs/stonehenge.txt',
            "start_pos": [-0.47, -0.7, 0.1],
            "end_pos": [0.12, 0.51, 0.16],
            "astar": True,
            "astar_kernel": 5,
            "subsample": 1,
            }

    experiment_name = cfg['experiment_name']
    renderer = get_nerf(cfg['nerf_config_file'], need_render=False)

    mesh = get_mesh(renderer)
    start_pos = torch.tensor(cfg['start_pos'])
    end_pos = torch.tensor(cfg['end_pos'])

    rrt = RRT(start_pos, end_pos, mesh)

    basefolder = "experiments" / pathlib.Path(experiment_name)
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "train").mkdir()

    print("created", basefolder)
    (basefolder / 'cfg.json').write_text(json.dumps(cfg))


    traj = MinSnap(rrt.waypoints, subsample=cfg['subsample'])
    traj.solve()

    traj.save_data(basefolder / "train" / "0.json")
    exit()


    # trimesh.repair.fix_inversion(mesh, multibody=True)
    # mesh = pymesh.form_mesh(vertices, faces)

    # mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxels )

    points = coods.reshape(-1,3).detach().numpy()
    inside = mesh.contains(points)
    distances = trimesh.proximity.ProximityQuery(mesh).signed_distance(points)

    print(sum(distances < 0))
    print(sum(inside))

    # mesh.bounding_box.cornejoo
    print(trimesh.bounds.corners(mesh.bounding_box_oriented.bounds))
    print(distances)

    # mesh.show()
    


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


