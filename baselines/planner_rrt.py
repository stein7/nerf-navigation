
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
    def __init__(self, start_state, end_state, mesh, cfg):
        self.start_state = start_state
        self.end_state = end_state
        self.mesh = mesh

        self.cfg = cfg
        self.goal_prob = cfg['goal_prob']
        self.max_step_distance = cfg['max_step_distance']
        self.robot_radius = cfg['robot_radius']
        self.line_step = cfg['rrt_line_step']

        # map from child to parent point
        self.graph = { self.hashable(start_state) : None }
    
        while True:
            # pick random point (pick goal point with prob)
            use_final_goal = np.random.random() < self.goal_prob
            if use_final_goal:
                new_point = self.end_state
                print("using final", new_point)
            else:
                # xy from -1 to 1, z from -0.1 to 0.9
                new_point = np.array([2,2,0.5]) * np.random.random((3)) - np.array([1,1,0.1])
                print("using random", new_point)

            if self.in_colision(new_point):
                continue
                print("in collision")

            # find closest point in tree
            closest = self.find_closest(new_point)
            print("found closest", closest)

            if self.line_step:
                successful = self.subdivide_line(new_point, closest)
            else:
                successful = self.add_closest(new_point, closest)

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

    def add_closest(self, target, tree_node):
        normal = target - tree_node
        distance = np.linalg.norm(normal)
        if distance == 0:
            return False
        normal = normal/distance

        if distance < self.max_step_distance:
            self.graph[self.hashable(target)] = self.hashable(tree_node)
            return True
        else:
            next_point = tree_node + normal * self.max_step_distance
            if not self.in_colision(next_point):
                self.graph[self.hashable(next_point)] = self.hashable(tree_node)
        return False

    def subdivide_line(self, target, tree_node):
        normal = target - tree_node
        distance = np.linalg.norm(normal)
        if distance == 0:
            return False
        normal = normal/distance

        prev_point = self.hashable(tree_node)

        # goes from the tree_node (exclusive) to the target (inclusive)
        nodes = int(distance//self.max_step_distance)
        for point in reversed( np.linspace(target, tree_node, nodes, endpoint=False) ):
            print("checking", point)
            if self.in_colision(point):
                return False
                print("colised")
            tuple_point = self.hashable(point)
            self.graph[tuple_point] = prev_point
            prev_point = tuple_point

        print("successgul")
        return True

    # def find_closest(self, point):
    #     # horribly slow etc
    #     min_distance = float("inf")
    #     min_point = None
    #     for node in self.graph.keys():
    #         distance = np.sum( (point - np.array(node))**2 )
    #         if distance < min_distance:
    #             min_distance = distance
    #             min_point = node

    #     return np.array(min_point)

    def find_closest(self, point):
        graph_points = np.array(list(self.graph.keys()))
        distances = np.linalg.norm(graph_points - point, axis = -1)
        index = np.argmin(distances)

        return np.array(graph_points[index])


    def in_colision(self, point):
        distances = trimesh.proximity.ProximityQuery(self.mesh).signed_distance(point.reshape(1,3))

        # how big is the robot # negative is outside
        if distances[0] < - self.robot_radius:
            return False

        return True



    
    def get_distance_to_go(self, state):
        pass

def get_mesh(renderer, points_per_side = 40):
    nerf = renderer.get_density

    side = points_per_side
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


def run_many():
    changes = [{"experiment_name":"random_rrt_stonehenge_0",
                    "start_pos": [0.55, -0.02, 0.05],
                    "end_pos": [-0.72, 0.17, 0.19],}, 

                {"experiment_name":"random_rrt_stonehenge_1",
                    "start_pos": [-0.12, -0.92, 0.05],
                    "end_pos": [-0.1, 0.8, 0.19],}, 

                {"experiment_name":"random_rrt_stonehenge_2",
                    "start_pos": [-0.72, -0.75, 0.1],
                    "end_pos": [0.51, 0.48, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_3",
                    "start_pos": [-0.42, -0.75, 0.1],
                    "end_pos": [0.21, 0.48, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_4",
                    "start_pos": [-0.12, -0.75, 0.1],
                    "end_pos": [-0.09, 0.48, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_5",
                    "start_pos": [0.18, -0.75, 0.1],
                    "end_pos": [-0.39, 0.48, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_6",
                    "start_pos": [0.48, -0.75, 0.1],
                    "end_pos": [-0.69, 0.48, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_7",
                    "start_pos": [0.48, -0.42, 0.1],
                    "end_pos": [-0.71, 0.83, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_8",
                    "start_pos": [-0.72, -0.12, 0.1],
                    "end_pos": [0.49, 0.83, 0.16],}, 

                {"experiment_name":"random_rrt_stonehenge_9",
                    "start_pos": [-0.72, -0.42, 0.1],
                    "end_pos": [0.49, 0.23, 0.16],}, ]

    for change in changes:
        cfg = { "experiment_name": None,
                "nerf_config_file": 'configs/stonehenge.txt',
                "start_pos": None,
                "end_pos": None,
                "mesh_points_per_side": 40,
                'rrt_line_step': False,
                'goal_prob': 0.05,
                'max_step_distance': 0.10,
                'robot_radius': 0.05,
                "minsnap_subsample": 1,
                "waypoint_dt": 0.1,
                }

        print(change)
        cfg.update(change)
        print(cfg)
        run_planner(cfg)


def main():
    cfg = { "experiment_name": "rrt_stonehenge_compare1",
            "nerf_config_file": 'configs/stonehenge.txt',
            "start_pos": [-0.47, -0.7, 0.1],
            "end_pos": [0.12, 0.51, 0.16],
            "mesh_points_per_side": 40,
            'rrt_line_step': False,
            'goal_prob': 0.05,
            'max_step_distance': 0.10,
            'robot_radius': 0.05,
            "minsnap_subsample": 1,
            "waypoint_dt": 0.1,
            }

    run_planner(cfg)

def run_planner(cfg):

    experiment_name = cfg['experiment_name']
    renderer = get_nerf(cfg['nerf_config_file'], need_render=False)

    mesh = get_mesh(renderer)
    start_pos = np.array(cfg['start_pos'])
    end_pos = np.array(cfg['end_pos'])

    basefolder = "experiments" / pathlib.Path(experiment_name)
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "train").mkdir()

    print("created", basefolder)
    (basefolder / 'cfg.json').write_text(json.dumps(cfg))

    rrt = RRT(start_pos, end_pos, mesh, cfg)
    traj = MinSnap(rrt.waypoints, subsample=cfg['minsnap_subsample'], nerf=renderer.get_density, waypoint_dt= cfg['waypoint_dt'])
    traj.solve()

    traj.save_data(basefolder / "train" / "0.json")


def testing():
    renderer = get_nerf('configs/stonehenge.txt', need_render=False)
    mesh = get_mesh(renderer)
    mesh.show()


if __name__ == "__main__":
    # testing()
    # main()
    run_many()

