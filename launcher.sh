#!/bin/bash
#SBATCH --job-name=pretrain_ndh_r2r
#SBATCH --output=logs/%j-pretrain_ndh_r2r.log
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=long
#SBATCH --exclude=rosie,cortana,vincent,calculon,droid,neo,irona

host="$(hostname)"
SLURM_INFO="JOB ID: ${SLURM_JOB_ID} JOB NAME: ${SLURM_JOB_NAME} NODE: ${host} NO OF GPUS: ${SLURM_GPUS} QUEUE: ${SLURM_JOB_PARTITION} QOS: ${SLURM_JOB_QOS}"
echo ${SLURM_INFO}
set -x
nvidia-docker run --ipc=host --volume `pwd`:/root/mount/Matterport3DSimulator --volume ~/data/visitron:/root/mount/Matterport3DSimulator/srv ayshrv/mattersim:visitron bash run_scripts_new/pretrain/pretrain_ndh_r2r.sh multi-gpu-ddp "${SLURM_INFO}"
