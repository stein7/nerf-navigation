---
abstract: Neural Radiance Fields (NeRFs) have recently emerged as a powerful paradigm for the representation of natural, complex 3D scenes. NeRFs represent continuous volumetric density and RGB values in a neural network, and generate photo-realistic images from unseen camera viewpoints through ray tracing.  We propose an algorithm for navigating a robot through a 3D environment represented as a NeRF using only an on-board RGB camera for localization.  We assume the NeRF for the scene has been pre-trained offline, and the robot's objective is to navigate through unoccupied space in the NeRF to reach a goal pose.  We introduce a trajectory optimization algorithm that avoids collisions with high-density regions in the NeRF based on a discrete time version of differential flatness that is amenable to constraining the robot's full pose and control inputs.  We also introduce an optimization based filtering method to estimate 6DoF pose and velocities for the robot in the NeRF given only an onboard RGB camera.  We combine the trajectory planner with the pose filter in an online replanning loop to give a vision-based robot navigation pipeline.  We present simulation results with a quadrotor robot navigating through a jungle gym environment, the inside of a church, and Stonehenge using only an RGB camera. We also demonstrate an omnidirectional ground robot navigating through the church, requiring it to reorient to fit through the narrow gap.
---

<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/5JjWpv9BaaE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->


<!-- <p align="center"> -->
<div class="aspect-ratio" align="center" style="position: relative; width: 100%; height: 0; padding-bottom: 75%;">
<iframe width="560" height="315" style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="https://www.youtube.com/embed/5JjWpv9BaaE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<!-- </p> -->



<div class="d-none d-md-block abstract">
	<h4> Abstract </h4>
	{{page.abstract}}
</div>
<a class="d-block d-md-none" data-toggle="collapse" data-target="#collapseExample" aria-expanded="false" aria-controls="collapseExample"><b>[<u>abstract</u>]</b></a>
<div class="collapse" id="collapseExample">
  <div class="card card-body abstract">
    {{page.abstract}}
  </div>
</div>

#### Trajectory Optimizer
<div class="row justify-content-center">
  <div class="col-8">
    <img src="assets/media/path_planning.gif" class="img-fluid mb-4" alt="Demonstration of our proposed NeRF-based trajectory optimizer. The drone plans a path which avoids collision and is dynamically feasible.">
  </div>
</div>

We propose a trajectory optimizer which uses a NeRF representation of the environment to encode obstacle geometry. As shown above, we initialize our planner with an A* search, and then use gradient descent to refine the trajectory to one which is dynamically feasible and minimizes both collision and control effort.

#### State Estimator

<div class="row justify-content-center">
  <div class="col-8">
    <img src="assets/media/estimator.gif" class="img-fluid mb-4" alt="Demonstration of the proposed NeRF-based state estimator. The mean state estimate converges quickly to a value where the ground truth and simulated images (via neural rendering) match closely.">
  </div>
</div>

We also propose a NeRF-based recursive state estimator, which uses the neural rendering process as a differentiable measurement model. Our method shows fast convergence to the true state (shown above, as the transparent "estimated" image, generated via neural rendering, converges to the opaque ground truth image), and is able to estimate the full robot state, including both 6D pose as well as linear and angular rates.

#### MPC Controller

<div class="row justify-content-center">
  <div class="col-8">
    <img src="assets/media/mpc.gif" class="img-fluid mb-4" alt="A drone flies through Stonehenge. Using our proposed method, it can replan online to account for unexpected disturbances and avoid collision.">
  </div>
</div>

We can combine the trajectory optimiser and state estimator into a controller capable of localising and dynamically re-optimising the trajectory to reject noise and disturbances.
