# nerf-nav
#### A Navigation Pipeline using PyTorch and NeRFs
### [Project](https://mikh3x4.github.io/nerf-navigation/) | [Video](https://youtu.be/5JjWpv9BaaE) | [Paper](https://arxiv.org/abs/2110.00168)

[Vision-Only Robot Navigation in a Neural Radiance World](https://mikh3x4.github.io/nerf-navigation/)  
 [Michal Adamkiewicz](https://profiles.stanford.edu/michal-adamkiewicz)\*<sup></sup>,
 [Timothy Chen](https://msl.stanford.edu/people/timchen)\*<sup></sup>,
 [Adam Caccavale](https://msl.stanford.edu/people/adamcaccavale)<sup></sup>,
 [Rachel Gardner](https://rachel-gardner.com/)<sup></sup>,
 [Preston Culbertson ](https://web.stanford.edu/~pculbert/)<sup></sup>,
 [Jeannette Bohg](https://web.stanford.edu/~bohg/)<sup></sup>, 
 [Mac Schwager](https://web.stanford.edu/~schwager/)<sup></sup> <br>
  \*denotes equal contribution

<p align="center">
    <img src="assets/drone_headline.jpg"/>
</p>

## What is NeRF-Nav?
NeRFs have recently emerged as a powerful paradigm for the representation of natural, complex 3D scenes. NeRFs represent continuous volumetric density and RGB values in a neural network, and generate photo-realistic images from unseen camera viewpoints through ray tracing. We propose an algorithm for navigating a robot through a 3D environment represented as a NeRF using only an on-board RGB camera for localization. We assume the NeRF for the scene has been pre-trained offline, and the robot's objective is to navigate through unoccupied space in the NeRF to reach a goal pose. We introduce a trajectory optimization algorithm that avoids collisions with high-density regions in the NeRF based on a discrete time version of differential flatness that is amenable to constraining the robot's full pose and control inputs. We also introduce an optimization based filtering method to estimate 6DoF pose and velocities for the robot in the NeRF given only an onboard RGB camera. We combine the trajectory planner with the pose filter in an online replanning loop to give a vision-based robot navigation pipeline. We present simulation results with a quadrotor robot navigating through a jungle gym environment, the inside of a church, and Stonehenge using only an RGB camera. We also demonstrate an omnidirectional ground robot navigating through the church, requiring it to reorient to fit through the narrow gap.

## Setup 

### Option 1: Using pip

In a new `conda` or `virtualenv` environment, run

```bash
pip install -r requirements.txt
```

### Option 2: Using conda

Use the provided `environment.yml` file to install the dependencies into an environment named `nerf` (edit the `environment.yml` if you wish to change the name of the `conda` environment).

```bash
conda env create
conda activate nerf
```

Note that it is also necessary to install [LieTorch] (https://github.com/princeton-vl/lietorch) in order to run our state estimator.

You will also need to install [Blender] (https://www.blender.org/) to use its scripting feature, which allows our code to talk to an interactive simulation environment.

### Deploy

Once everything is setup, to run experiments, first edit `config/stonehenge_nav.yml` to specify your own parameters. Make sure your NeRF model is also trained using [A PyTorch re-implementation] (https://github.com/krrish94/nerf-pytorch), which is the NeRF implementation our code runs on.

First, open Blender and the scripting tab, and open `visualize.py`. NOTE: Be sure to run Blender from terminal in order to manually break out of the loop. Then, run the file. At this point, the simulation will be listening to any poses given by our navigation code. We assume that the incoming images from Blender are 800x800, which are resized to 400x400 for the filter. As a result, the trained NeRF should also be trained on 800x800 images with the parameter `half_res=True`.

The navigation script can be invoked by running
```bash
python simulate.py --config config/stonehenge_nav.yml
```

### File Structure
It is best to understand a bit more about how the data and outputs are organized. In the root directory, there is only 1 file pertinent to the navigation, and that is `simulate.py`. The rest of the files are there for training and evaluating NeRFs, just like in [A PyTorch re-implementation] (https://github.com/krrish94/nerf-pytorch). 


`simulate.py` is the entrance to our navigation code. This is where the NeRF model is loaded, the MPC parameters are loaded, integration between the planner and filter, and MPC execution.

`-sim_img_cache` is where Blender updates images and receives poses from the navigation code.

`-paths` is where navigation-specific logging data is stored.

`-nerf` is where NeRF-specific utilities are stored. This is unchanged from the underlying NeRF implementation.

`-logs` is where the trained NeRF weights go.

`-cache` is where the data used to train the NeRF goes.

`-config` is where the config files for both training and navigation go. Note that the navigation config files are a superset of the training files.

`-nav` contains the navigation-specific code. In it, we will find:

```
-agent_helpers.py
-estimator_helpers.py
-mpc_utils.py
-quad_helpers.py
-quad_plot.py
```

`agent_helpers.py` contains the dynamics model of a drone, and also interfaces with our Blender simulation environment.

`estimator_helpers.py` contains the utilities for the filter.

`mpc_utils.py` contains helper functions for the MPC integration.

`quad_helpers.py` and `quad_plot.py` contain helper functions for the planner.

### Logs
In `logs` folder, a folder will automatically be generated storing your rendered test images and neural network weights, along with some text files indicating the config used to train the model (Very important when sharing models with others!).

### Configs
In `configs` folder contains the config file used to train a particular NeRF. It is highly recommended to take a look at the example config files in order to understand how your model will behave during training. Some parameters that are particularly important if you decide to copy and paste the example config files are `expname, datadir, dataset_type, white_bkgd, half_res` which determine the experiment's name and corresponding name of the log file in `logs`, the directory in which you stored the training data, where you got your dataset from (e.g., Blender), whether or not the NeRF should be trained on images with white backgrounds, and whether you want your model to train on training images at half resolution.

NOTE: `white_bkgd` primarily applies to Blender datasets that have transparent images so that setting `white_bkgd=True` will allow the NeRF to render properly. If your images have solid background colors, set this parameter to False.

NOTE: Setting `half_res` to True will also cause the NeRF model to render at half resolution.

### Data
The `data` folder is separated into real-life training data `nerf_llff_data` and synthetic (e.g., Blender) data in `nerf_synthetic`. However, the structure within both is the same. Within each scenes folder, there MUST HAVE 3 folders `test`, `train`, and `val` containing the corresponding images, and their respective ground truth poses under `transforms_....json`. It is recommended to look at the `.json` file to see camera instrinsic parameters that the file should provide beside poses.

## Blender Specific
We will eventually provide a script where you can generate these three folders and pose files after loading a scene or object into Blender.

In the meantime, the `.json` file is structured as a dictionary:
```
{
  "Far": ...,   #Far Bound
  "Near": ...,  #Near Bound
  "camera_angle_x: ..., #Horizontal FOV
  "frames": ...
}
```
where `"frames"` is a list of dictionaries (one for each image) containing the file path to the image and its corresponding ground-truth pose as follows:

```
{
  "transform_matrix": ...,   #Pose in SE3
  "file_path": "./{test,train,val}/img_name"  #File path
}
```

## LICENSE

`nerf-pytorch` is available under the [MIT License](https://opensource.org/licenses/MIT). For more details see: [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).
