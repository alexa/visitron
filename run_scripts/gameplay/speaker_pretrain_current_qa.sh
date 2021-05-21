cpu="python"
single_gpu="python"
multi_gpu_data_parallel="python"
multi_gpu_dist_data_parallel="python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --node_rank 0"

case $1 in
    cpu)
        setting="CUDA_VISIBLE_DEVICES=-1 $cpu"
        slurm_info=$2
        ;;
    single-gpu)
        setting="CUDA_VISIBLE_DEVICES=$2 $single_gpu"
        slurm_info=$3
        ;;
    multi-gpu-dp)
        setting=$multi_gpu_data_parallel
        slurm_info=$2
        ;;
    multi-gpu-ddp)
        setting=$multi_gpu_dist_data_parallel
        slurm_info=$2
        ;;
    *)
    echo Unknown setting, Options: cpu, single-gpu \$GPU_ID, multi-gpu-dp, multi-gpu-ddp. Optionally add SLURM INFO after this.
    exit 1
    ;;
esac

file="tasks/viewpoint_select/train_gameplay.py"

arguments="
--img_feat_dir srv/img_features
--img_feature_file ResNet-152-imagenet.tsv
--data_dir srv/task_data/NDH/data
--output_dir srv_2/results/gameplay/speaker_pretrain-current_qa-$1
--history nav_q_oracle_ans
--speaker_only
--max_seq_length 512
--img_feature_dim 2054
--lstm_img_feature_dim 2048
--per_gpu_train_batch_size 8
--per_gpu_eval_batch_size 8
--learning_rate 5e-05
--weight_decay 0.05
--num_iterations 20000
--warmup_steps 0
--drop_out 0.3
--logging_steps 200
--eval_logging_steps 500
--save_steps 100
--seed 88
--num_workers 0
"
# --evaluate_during_training

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
