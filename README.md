# nerf-nav
#### A Navigation Pipeline using PyTorch and NeRFs
### [Project](https://mikh3x4.github.io/nerf-navigation/) | [Video](https://youtu.be/5JjWpv9BaaE) | [Paper](https://arxiv.org/abs/2110.00168)

[Vision-Only Robot Navigation in a Neural Radiance World](https://mikh3x4.github.io/nerf-navigation/)  
 [Michal Adamkiewicz](https://profiles.stanford.edu/michal-adamkiewicz)\*<sup></sup>,
 [Timothy Chen](https://msl.stanford.edu/people/timchen)\*<sup></sup>,
 [Adam Caccavale](https://msl.stanford.edu/people/adamcaccavale)\<sup></sup>,
 [Rachel Gardner](https://rachel-gardner.com/)<sup></sup>,
 [Preston Culbertson ](https://web.stanford.edu/~pculbert/)<sup></sup>,
 [Jeannette Bohg](https://web.stanford.edu/~bohg/)<sup></sup> 
 [Mac Schwager](https://web.stanford.edu/~schwager/)<sup></sup> <br>
  \*denotes equal contribution

<p align="center">
    <img src="assets/pipeline.jpg"/>
</p>

## How to train your NeRF super-quickly!

To train a "full" NeRF model (i.e., using 3D coordinates as well as ray directions, and the hierarchical sampling procedure), first setup dependencies. 

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

### Run training!

Once everything is setup, to run experiments, first edit `config/lego.yml` to specify your own parameters.

The training script can be invoked by running
```bash
python train_nerf.py --config config/lego.yml
```

### Optional: Resume training from a checkpoint

Optionally, if resuming training from a previous checkpoint, run
```bash
python train_nerf.py --config config/lego.yml --load-checkpoint path/to/checkpoint.ckpt
```

### Optional: Cache rays from the dataset

An optional, yet simple preprocessing step of caching rays from the dataset results in substantial compute time savings (reduced carbon footprint, yay!), especially when running multiple experiments. It's super-simple: run
```bash
python cache_dataset.py --datapath cache/nerf_synthetic/lego/ --halfres False --savedir cache/legocache/legofull --num-random-rays 8192 --num-variations 50
```

This samples `8192` rays per image from the `lego` dataset. Each image is `800 x 800` (since `halfres` is set to `False`), and `500` such random samples (`8192` rays each) are drawn per image. The script takes about 10 minutes to run, but the good thing is, this needs to be run only once per dataset.

> **NOTE**: Do NOT forget to update the `cachedir` option (under `dataset`) in your config (.yml) file!

## Render fun videos (from a pretrained model)

Once you've trained your NeRF, it's time to use that to render the scene. Use the `eval_nerf.py` script to do that. For the `lego-lowres` example, this would be
```bash
python eval_nerf.py --config pretrained/lego-lowres/config.yml --checkpoint pretrained/lego-lowres/checkpoint199999.ckpt --savedir cache/rendered/lego-lowres
```

You can create a `gif` out of the saved images, for instance, by using [Imagemagick](https://imagemagick.org/).
```bash
convert cache/rendered/lego-lowres/*.png cache/rendered/lego-lowres.gif
```

This should give you a gif like this.

<p align="center">
    <img src="assets/lego-lowres.gif">
</p>

**Pretrained models**: Pretrained models for the following scenes are available in the `pretrained` directory (all of them are currently lowres). I will continue adding models herein.
```
# Synthetic (Blender) scenes
chair
drums
hotdog
lego
materials
ship

# Real (LLFF) scenes
fern
```

## LICENSE

`nerf-pytorch` is available under the [MIT License](https://opensource.org/licenses/MIT). For more details see: [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).
