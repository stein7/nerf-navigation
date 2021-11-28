
import sys
sys.path.append('.')
# run from toplevel directory or imports will fail

import numpy as np
import torch

import trimesh
from skimage import measure

from load_nerf import get_nerf

torch.manual_seed(0)
np.random.seed(0)


class RTT:

    def __init__(self, starting_state):
        pass

    def get_distance_to_go(self, state):
        pass


def testing():
    pass
    renderer = get_nerf('configs/stonehenge.txt')
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
    print(voxels)

    voxels[0,:,:] = 0
    voxels[:,0,:] = 0
    voxels[:,:,0] = 0

    voxels[-1,:,:] = 0
    voxels[:,-1,:] = 0
    voxels[:,:,-1] = 0

    vertices, faces, normals, _ = measure.marching_cubes(voxels.detach().numpy(), spacing=(spacing, spacing, spacingz))
    vertices += np.array([ linspace[0] - spacing/2, linspace[0] - spacing/2, linspacez[0] - spacingz/2] )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    # trimesh.repair.fix_inversion(mesh, multibody=True)
    # mesh = pymesh.form_mesh(vertices, faces)

    # mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxels )

    points = coods.reshape(-1,3).detach().numpy()
    distances = mesh.contains(points)

    # mesh.bounding_box.cornejoo
    print(trimesh.bounds.corners(mesh.bounding_box_oriented.bounds))
    print(distances)

    mesh.show()
    


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


