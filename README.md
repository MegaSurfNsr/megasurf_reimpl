<p align="center">
    <h1 align="center">Re-implementation</h1>
    <h1 align="center">MegaSurf: Efficient and Robust Sampling Guided Neural Surface Reconstruction Framework of Scalable Large Scene</h1>
    <h3 align="center"><a href="https://megasurfnsr.github.io/">Project Page</a> </h3>
</p>

# About

Due to company licensing reasons, the code cannot be taken out. I reproduced most of the content in the article myself based on the [SDFStudio](https://autonomousvision.github.io/sdfstudio/) and the experiment results are consistent with that shown in the paper. It should be noted that in this repo: 1. The geometry prior directly uses the output of PGSR. 2. Parameters will need to be adjusted later to accelerate training speed.
# Updates

**2024.10.20**: First upload.

**2024.10.20**: Project page. [MegaSurf](https://megasurfnsr.github.io/).


# Quickstart
## The environment setup is very similar to that of SDFstudio.
## 1. Installation: Setup the environment

### Prerequisites

CUDA must be installed on the system. This library has been tested with version 11.3. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Create environment

SDFStudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name sdfstudio -y python=3.8
conda activate sdfstudio
python -m pip install --upgrade pip
```

### Dependencies

Install pytorch with CUDA (this repo has been tested with CUDA 11.3) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing SDFStudio

```bash
git clone https://github.com/autonomousvision/sdfstudio.git
cd sdfstudio
pip install --upgrade pip setuptools
pip install -e .
```

## 2. Train your first model
Here we provide a [demo data(3.81G)](https://drive.google.com/file/d/1yib6I2T-bb_60uoZZrDRAi1ImblMfmyC/view) of Urbanscene3D-residence and [reconstruction result(ply)](https://drive.google.com/file/d/1L6mRkcYXxWWyRWJgwyGSKtlYidVA1St5/view?usp=sharing) by this repo.

The following will train a Megasurf model,

```bash
# Train model on the demo dataset residence49. I took a shortcut here and modified bakedangelo directly without adding extra modules for SDFstudio.
python ./scripts/train.py bakedangelo --machine.num-gpus 1 --pipeline.model.level-init 8 --trainer.steps-per-eval-image 5000 --trainer.max-num-iterations 300010 --trainer.steps-per-save 10000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.datamanager.eval-num-rays-per-batch 512 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.background-color white --pipeline.model.sdf-field.bias 0.1 --pipeline.model.sdf-field.inside-outside False --pipeline.model.background-model grid --pipeline.model.steps_per_level 2000 --vis tensorboard --output-dir /mnt/data3/yswang2024_data3/megasurf_output --experiment-name your_exp_names_here nerfstudio-data --data your_dataset_here_which_contains_json_file --downscale-factor 1 --use_all_train_images True --center_poses False --orientation_method none

# Extract the mesh
python ./scripts/extract_mesh.py --load-config exp_save_dir/config.yml --output-path save_dir/mesh.ply

# You can use the tools in SDFstudio like tensorboard to monitor the training process.
tensorboard --logdir exp_path/ --bind_all --port 6088
# Then, open your local browser and enter the address XXX.XXX.XXX.XXX:6088 to access the TensorBoard content. 
```


# Built On
<a href="https://github.com/autonomousvision/sdfstudio">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/autonomousvision/sdfstudio/blob/master/media/sdf_studio_4.png" />
<!-- /pypi-strip -->
    <img alt="tyro logo" src="https://github.com/autonomousvision/sdfstudio/blob/master/media/sdf_studio_4.png" width="150px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- A Unified Framework for Surface Reconstruction
- Developed by [sdfstudio team](https://autonomousvision.github.io/sdfstudio/)
- 
<a href="https://github.com/nerfstudio-project/nerfstudio">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/en/latest/_images/logo.png" />
<!-- /pypi-strip -->
    <img alt="tyro logo" src="https://docs.nerf.studio/en/latest/_images/logo.png" width="150px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- A collaboration friendly studio for NeRFs
- Developed by [nerfstudio team](https://github.com/nerfstudio-project)

<a href="https://github.com/brentyi/tyro">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://brentyi.github.io/tyro/_static/logo-dark.svg" />
<!-- /pypi-strip -->
    <img alt="tyro logo" src="https://brentyi.github.io/tyro/_static/logo-light.svg" width="150px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- Easy-to-use config system
- Developed by [Brent Yi](https://brentyi.com/)

<a href="https://github.com/KAIR-BAIR/nerfacc">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png" />
<!-- /pypi-strip -->
    <img alt="tyro logo" src="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" width="250px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- Library for accelerating NeRF renders
- Developed by [Ruilong Li](https://www.liruilong.cn/)

# Citation


```bibtex
@inproceedings{wang2024megasurf,
  title={MegaSurf: Scalable Large Scene Neural Surface Reconstruction},
  author={Wang, Yusen and Zhou, Kaixuan and Zhang, Wenxiao and Xiao, Chunxia},
  booktitle={ACM Multimedia 2024}
}
```
