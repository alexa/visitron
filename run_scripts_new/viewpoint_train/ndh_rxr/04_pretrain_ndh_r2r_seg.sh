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

file="tasks/viewpoint_select/train.py"

arguments="
--img_feat_dir srv/img_features
--img_feature_file Learned-Seg.tsv
--data_dir srv/task_data/NDH/data
--model_name_or_path srv_2/results/pretrain/pretrain_masked_lm_1-in-36-viewpoint_ndh_r2r-TAR-front-multi-gpu-ddp/checkpoints/checkpoint-30000
--output_dir srv_2/results/viewpoint-2/ndh_rxr-pretrain_ndh_r2r-semantic_seg
--add_rxr_data
--feedback_method sample
--path_type planner_path
--max_seq_length 768
--img_feature_dim 2054
--lstm_img_feature_dim 42
--per_gpu_train_batch_size 4
--per_gpu_eval_batch_size 4
--learning_rate 5e-05
--weight_decay 0.05
--num_iterations 50000
--warmup_steps 0
--drop_out 0.3
--logging_steps 10
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
