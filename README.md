# VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator

[VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator](https://arxiv.org/abs/2105.11589)

Ayush Shrivastava, Karthik Gopalakrishnan, Yang Liu, Robinson Piramuthu, Gokhan Tür, Devi Parikh, Dilek Hakkani-Tür

NAACL 2021, Visually Grounded Interaction and Language (ViGIL) Workshop

![VISITRON](visitron.png)

Citation:
```
@inproceedings{visitron,
  title={VISITRON: Visual Semantics-aligned Interactively Trained Object-Navigator},
  author={Ayush Shrivastava, Karthik Gopalakrishnan, Yang Liu, Robinson Piramuthu, Gokhan Tür, Devi Parikh, Dilek Hakkani-T\"{u}r},
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
This codebase uses the Matterport3D Simulator. Detailed instructions on how to setup the simulator and how to preprocess the Matterport3D data for faster simulator performance are present here: [Matterport3DSimulator_README](https://github.com/mmurray/cvdn/blob/master/README_Matterport3DSimulator.md). We provide the docker setup for ease of setup for the simulator.

We assume that the Matterport3D is present at `$MATTERPORT_DATA_DIR` which can be set using:
```
export MATTERPORT_DATA_DIR=<PATH_TO_MATTERPORT_DATASET>
```

### Docker setup

Build the docker image:
```
docker build -t mattersim:visitron .
```

To run the docker container, mounting the codebase and the Matterport3D dataset, use:
```
nvidia-docker run -it --ipc=host --cpuset-cpus="$(taskset -c -p $$ | cut -f2 -d ':' | awk '{$1=$1};1')" --volume `pwd`:/root/mount/Matterport3DSimulator --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans,readonly mattersim:visitron
```

### Task data setup
Our approach is pretrained on NDH and R2R, and then finetuned on NDH and RxR. Download these task data as follows.

#### NDH, R2R data :

```
mkdir -p srv/task_data
bash scripts/download_ndh_r2r_data.sh
```

#### RxR data :

Refer to [RxR repo](https://github.com/google-research-datasets/RxR#dataset-download) for its setup and copy the data to `srv/task_data/RxR/data` folder.

### Image features

### Oscar setup



-
 - use docker setup
 -

# Setup

## Setup

### Setup

#### Setup

##### Setup


sdfds







Contains code to train VISITRON, an [Oscar](https://github.com/microsoft/Oscar)-based agent
for NDH/[CVDN](https://github.com/mmurray/cvdn) task.
Work in progress.


## License

This library is licensed under the MIT-0 License. See the LICENSE file.


## Setup

This repo uses
- Bottom-Up Attention repo. Create a folder called `/bottom-up` and copy the code from this repo (https://github.com/peteanderson80/bottom-up-attention/tree/ec7422aa6b672ff8c8a3594805cbe269cbf29723).
- Transformers repo used for running Oscar model. Install it from here (https://github.com/huggingface/transformers/tree/067923d3267325f525f4e46f357360c191ba562e) to `/tasks/viewpoint_select/oscar/transformers_src`.

### R4R Data Setup

Create a folder `/generate_r4r` and copy [r4r_generate_data.py](https://github.com/google-research/google-research/blob/master/r4r/r4r_generate_data.py) and [graph_utils.py](https://github.com/google-research/google-research/blob/master/r4r/graph_utils.py) to it. Refer to [this repo](https://github.com/google-research/google-research/tree/master/r4r) for other details about R4R.


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

- generate r4r

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


