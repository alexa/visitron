# VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator

[VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator](https://arxiv.org/abs/2105.11589)

Ayush Shrivastava, Karthik Gopalakrishnan, Yang Liu, Robinson Piramuthu, Gokhan Tür, Devi Parikh, Dilek Hakkani-Tür

Accepted to NAACL 2021, Visually Grounded Interaction and Language (ViGIL) Workshop

![VISITRON](visitron.png)

Citation:
```
@inproceedings{visitron,
  title={VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator},
  author={Ayush Shrivastava, Karthik Gopalakrishnan, Yang Liu, Robinson Piramuthu, Gokhan T\"{u}r, Devi Parikh, Dilek Hakkani-T\"{u}r},
  booktitle={NAACL 2021, Visually Grounded Interaction and Language (ViGIL) Workshop},
  year={2021}
}
```

## Setup
Clone the repo using:
```
git clone --recursive https://github.com/alexa/visitron.git
```

### Matterport3D dataset and simulator
This codebase uses the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator). Detailed instructions on how to setup the simulator and how to preprocess the Matterport3D data for faster simulator performance are present here: [Matterport3DSimulator_README](https://github.com/mmurray/cvdn/blob/master/README_Matterport3DSimulator.md). We provide the docker setup for ease of setup for the simulator.

We assume that the Matterport3D is present at `$MATTERPORT_DATA_DIR` which can be set using:
```
export MATTERPORT_DATA_DIR=<PATH_TO_MATTERPORT_DATASET>
```

### Docker setup

Build the docker image:
```
docker build -t mattersim:visitron .
```

To run the docker container and mount the codebase and the Matterport3D dataset, use:
```
nvidia-docker run -it --ipc=host --cpuset-cpus="$(taskset -c -p $$ | cut -f2 -d ':' | awk '{$1=$1};1')" --volume `pwd`:/root/mount/Matterport3DSimulator --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans,readonly mattersim:visitron
```

### Task data setup
VISITRON is pretrained on [NDH](https://github.com/mmurray/cvdn) and [R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R), and then finetuned on NDH and [RxR](https://github.com/google-research-datasets/RxR). Download these task data as follows.

#### NDH, R2R data :

```
mkdir -p srv/task_data
bash scripts/download_ndh_r2r_data.sh
```

#### RxR data :

Refer to [RxR repo](https://github.com/google-research-datasets/RxR#dataset-download) for its setup and copy the data to `srv/task_data/RxR/data` folder.


### Pretraining data

Inside the docker container, run these commands to generate pretraining data.

For NDH, run
```
python scripts/generate_pretraining_data.py --dataset_to_use NDH --split train
```
For R2R, run
```
python scripts/generate_pretraining_data.py --dataset_to_use NDH --split train
```
The data gets saved to `srv/task_data/pretrain_data`.
By default, this script starts 8 multiprocessing threads to speed up its execution. `--start_job_index`, `--end_job_index` and `--global_total_jobs` can be changed to change the number of threads.



### Image features

Our pretraining approach requires object-level features from Faster RCNN concatenated with orientation features.

First, follow the setup from the [bottom-up repo](https://github.com/peteanderson80/bottom-up-attention) inside a docker container to install caffe. Note that the code from bottom-up repo requires python2.

Then, extract object-level features using
```
python2 scripts/precompute_bottom-up_features.py
```
You can use `--gpu_id` to parallelize the feature extraction process ovewr multiple GPUs.

Then, to concatenate orientation features to object-level features, use
```
python

Download ResNet features from [here](https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1).

TODO: ResNet-101-faster-rcnn-genome-worientation features

### Oscar setup
We use [Oscar](https://github.com/microsoft/Oscar) model as the backbone for our VISITRON model. Copy the Oscar pretrained weights from here.
TODO


## License

This library is licensed under the MIT-0 License. See the LICENSE file.


## Setup

This repo uses
- Bottom-Up Attention repo. Create a folder called `/bottom-up` and copy the code from this repo (https://github.com/peteanderson80/bottom-up-attention/tree/ec7422aa6b672ff8c8a3594805cbe269cbf29723).
- Transformers repo used for running Oscar model. Install it from here (https://github.com/huggingface/transformers/tree/067923d3267325f525f4e46f357360c191ba562e) to `/tasks/viewpoint_select/oscar/transformers_src`.


## How to run scripts in `run_scripts_new`

Use
```
bash run_scripts_new/viewpoint_train/ndh/01_pretrain_ndh.sh MODE
```
where `MODE` can be from [`cpu`, `single-gpu`, `multi-gpu-dp`, `multi-gpu-ddp`].
- Use `cpu` to train on CPU.
- Use `single-gpu` to train on a single GPU.
- Use `multi-gpu-dp` to train on all available GPUs using DataParallel.
- Use `multi-gpu-ddp` to train on 4 GPUs using DistributedDataParallel. Change `--nproc_per_node` in the script to specify no. of GPUs in DistributedDataParallel mode.



todos:

- README

- img_features
- bottom up

- scripts
    - features
    - pretraining data



--img_feat_dir srv/img_features
--img_feature_file ResNet-152-imagenet.tsv
--data_dir srv/task_data/NDH/data
--model_name_or_path srv/oscar_pretrained_models/base-vg-labels/ep_107_1192087
--output_dir srv_2/results/viewpoint-2/temp

exp_name=srv/results/viewpoint_select/ndh-no_pretraining


features used:
ResNet-152-imagenet.tsv
ResNet-101-faster-rcnn-genome-worientation


