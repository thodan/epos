
# EPOS: Estimating 6D Pose of Objects with Symmetries

This repository provides the source code and trained models of the 6D object pose estimation method presented in:

[Tomas Hodan](http://www.hodan.xyz), [Daniel Barath](http://web.eee.sztaki.hu/~dbarath/), [Jiri Matas](http://cmp.felk.cvut.cz/~matas/) <br>
**EPOS: Estimating 6D Pose of Objects with Symmetries**<br>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2020<br>
[PDF](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf) | [BIB](http://cmp.felk.cvut.cz/~hodanto2/data/hodan2020epos.bib) | [Video](https://www.youtube.com/watch?v=OXjG0YPqLnE) | [Project website](http://cmp.felk.cvut.cz/epos/)


Contents: [Setup](#setup) | [Usage](#usage) | [Pre-trained models](#pre-trained-models)


## <a name="setup"></a>1. Setup

The following sections will guide you in setting up the code on your machine. The code was developed and tested on Linux with GCC 8.3.0 (C++17 support is required). Please try to switch to this or a higher GCC version if you experience any compilation issues.

### 1.1 Python environment and dependencies

Create a conda environment and install dependencies:
```
conda create --name epos python=3.6.10
conda activate epos

conda install numpy=1.16.6
conda install tensorflow-gpu=1.12.0
conda install pyyaml=5.3.1
conda install opencv=3.4.2
conda install pandas=1.0.5
conda install tabulate=0.8.3
conda install imageio=2.9.0
conda install -c mjirik pypng=0.0.18
conda install -c conda-forge igl
conda install glog=0.4.0
```

To set environment variables, create file ```$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```, copy into it the following content, and adjust variables ```REPO_PATH```, ```STORE_PATH```, and ```BOP_PATH```:
```
#!/bin/sh

export REPO_PATH=/path/to/epos/repository  # Folder for the EPOS repository.
export STORE_PATH=/path/to/epos/store  # Folder for TFRecord files and trained models.
export BOP_PATH=/path/to/bop/datasets  # Folder for BOP datasets (bop.felk.cvut.cz/datasets).

export TF_DATA_PATH=$STORE_PATH/tf_data  # Folder with TFRecord files.
export TF_MODELS_PATH=$STORE_PATH/tf_models  # Folder with trained EPOS models.

export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/bop_renderer/build:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH

export LD_LIBRARY_PATH=$REPO_PATH/external/llvm/lib:$LD_LIBRARY_PATH
```

Re-activate the conda environment to load the environment variables:
```
conda activate epos
```

### 1.2 Cloning the repository

Download the code (including git submodules) to ```$REPO_PATH```:
```
git clone --recurse-submodules https://github.com/thodan/epos.git $REPO_PATH/
```

### 1.3 BOP renderer

The [BOP renderer](https://github.com/thodan/bop_renderer) is used to render the ground-truth label maps and can run off-screen on a server. The renderer depends on [OSMesa](https://www.mesa3d.org/osmesa.html) and [LLVM](https://llvm.org/) which can be installed as follows:
```
# The installation locations of OSMesa and LLVM:
export OSMESA_PREFIX=$REPO_PATH/external/osmesa
export LLVM_PREFIX=$REPO_PATH/external/llvm

mkdir $OSMESA_PREFIX
mkdir $LLVM_PREFIX
cd $REPO_PATH/external/bop_renderer/osmesa-install
mkdir build; cd build
bash ../osmesa-install.sh
```

Compile the renderer by:
```
cd $REPO_PATH/external/bop_renderer
mkdir build; cd build
export PYTHON_PREFIX=$CONDA_PREFIX
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### 1.4 Progressive-X

[Progressive-X](https://github.com/danini/progressive-x) is used to estimate the 6D object poses from 2D-3D correspondences. Make sure your GCC version supports C++17 and compile Progressive-X by:
```
cd $REPO_PATH/external/progressive-x/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### 1.5 BOP datasets

For inference/evaluation on the existing [BOP datasets](https://bop.felk.cvut.cz/datasets/), you need to download the base archives, 3D object models, and test images to folder ```$BOP_PATH```. For training, you need to download also the training images. You can also use your own dataset prepared in the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).


## <a name="usage"></a>2. Usage

All of the following scripts should be run from folder ```$REPO_PATH/scripts```.

### 2.1 Converting a dataset into a TFRecord file

First, create a list of images to include in the TFRecord file (examples are in [create_example_list.py](https://github.com/thodan/epos/blob/master/scripts/create_example_list.py)):
```
python create_example_list.py --dataset=<dataset> --split=<split> --split_type=<split_type>
```

A text file with the list is saved in ```$TF_DATA_PATH/example_lists```.

Then, create the TFRecord file (examples are in [create_tfrecord.py](https://github.com/thodan/epos/blob/master/scripts/create_tfrecord.py)):
```
python create_tfrecord.py --dataset=<dataset> --split=<split> --split_type=<split_type> --examples_filename=<examples_filename> --add_gt=<add_gt> --shuffle=<shuffle> --rgb_format=<rgb_format>
```

The TFRecord file is saved in ```$TF_DATA_PATH```.
A sample TFRecord with YCB-V test images used in the BOP Challenge 2019/2020 can be downloaded from [here](https://bop.felk.cvut.cz/media/data/epos_store/ycbv_test_targets-bop19.tfrecord).


### 2.2 Inference with a pre-trained model

Download and unpack a [pre-trained model](#pre-trained-models) into folder ```$TF_MODELS_PATH```.

Select a GPU and run the inference:
```
export CUDA_VISIBLE_DEVICES=0
python infer.py --model=<model_name>
```

where ```<model_name>``` is the name of the pre-trained model (e.g. ```ycbv-bop20-xc65-f64```).

The estimated poses are saved in the [format expected by the BOP Challenge 2019/2020](https://bop.felk.cvut.cz/challenges/bop-challenge-2020/#formatofresults) into folder ```$TF_MODELS_PATH/<model_name>/infer```. To save also visualizations of the estimated poses into folder ```$TF_MODELS_PATH/<model_name>/vis```, append the last command above with flag  ```--vis```.

### 2.3 Training your own model

First, create folder ```$TF_MODELS_PATH/<model_name>```, where ```<model_name>``` is a name of the model to be trained. Inside this folder, create file ```params.yml``` specifying parameters of the model. An example model for the YCB-V dataset can be downloaded from [here](https://bop.felk.cvut.cz/media/data/epos_store/ycbv-example-xc65-f64.zip).

Model weights pretrained on ImageNet and COCO datasets, which can be used to initialize training of your models, can be downloaded from here [here](https://bop.felk.cvut.cz/media/data/epos_store/imagenet-coco-xc65.zip) (extract the archive into ```$TF_MODELS_PATH/imagenet-coco-xc65```).

Select a GPU and launch the training:
```
export CUDA_VISIBLE_DEVICES=0
python train.py --model=<model_name>
```


### 2.4 Checking the training input

You can check the training data by visualizing the augmented images and the corresponding ground-truth object labels, fragment labels, and 3D fragment coordinates:

```
python check_train_input.py --model=<model_name>
```

The visualizations are saved in folder ```$TF_MODELS_PATH/<model_name>/check_train_input```.


## <a name="pre-trained-models"></a>3. Pre-trained models

Models evaluated in the [CVPR 2020 paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hodan_EPOS_Estimating_6D_Pose_of_Objects_With_Symmetries_CVPR_2020_paper.pdf) (Xception-65 backbone, trained for 2M iterations):

- 64 fragments per object:
[LM-O](https://bop.felk.cvut.cz/media/data/epos_store/lmo-cvpr20-xc65-f64.zip),
[T-LESS](https://bop.felk.cvut.cz/media/data/epos_store/tless-cvpr20-xc65-f64.zip),
[YCB-V](https://bop.felk.cvut.cz/media/data/epos_store/ycbv-cvpr20-xc65-f64.zip)
- 256 fragments per object:
[LM-O](https://bop.felk.cvut.cz/media/data/epos_store/lmo-cvpr20-xc65-f256.zip),
[T-LESS](https://bop.felk.cvut.cz/media/data/epos_store/tless-cvpr20-xc65-f256.zip),
[YCB-V](https://bop.felk.cvut.cz/media/data/epos_store/ycbv-cvpr20-xc65-f256.zip)


Models evaluated in the [BOP Challenge 2020](https://bop.felk.cvut.cz/challenges/bop-challenge-2020/) (Xception-65 backbone, trained only on the provided PBR images):

- 64 fragments per object:
[LM-O](https://bop.felk.cvut.cz/media/data/epos_store/lmo-bop20-xc65-f64.zip),
[T-LESS](https://bop.felk.cvut.cz/media/data/epos_store/tless-bop20-xc65-f64.zip),
[TUD-L](https://bop.felk.cvut.cz/media/data/epos_store/tudl-bop20-xc65-f64.zip),
[IC-BIN](https://bop.felk.cvut.cz/media/data/epos_store/icbin-bop20-xc65-f64.zip),
[ITODD](https://bop.felk.cvut.cz/media/data/epos_store/itodd-bop20-xc65-f64.zip),
[HB](https://bop.felk.cvut.cz/media/data/epos_store/hb-bop20-xc65-f64.zip),
[YCB-V](https://bop.felk.cvut.cz/media/data/epos_store/ycbv-bop20-xc65-f64.zip)
