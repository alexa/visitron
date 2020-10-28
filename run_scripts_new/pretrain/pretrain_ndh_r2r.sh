
arguments="
--img_feat_dir srv/img_features
--img_feature_file ResNet-101-faster-rcnn-genome-worientation-faster.pickle
--data_dir task_data/NDH/data
--model_name_or_path srv/oscar_pretrained_models/base-vg-labels/ep_107_1192087
--output_dir srv/results/pretrain/DDP-pretrain_masked_lm_1-in-36-viewpoint_ndh_r2r-DDP
--add_ndh_data
--add_r2r_data
--max_seq_length 768
--img_feature_dim 2054
--per_gpu_train_batch_size 4
--action_space 36
--learning_rate 5e-05
--weight_decay 0.05
--num_epochs 20
--warmup_steps 0
--drop_out 0.3
--logging_steps 10
--save_steps 100
--seed 88
--num_workers 8
"
# --evaluate_during_training

single_gpu="python"
multi_gpu_data_parallel="python"
multi_gpu_dist_data_parallel="python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --node_rank 0"

case $1 in
    single-gpu)
        setting="CUDA_VISIBLE_DEVICES=$2 $single_gpu"
        ;;
    multi-gpu-dp)
        setting=$multi_gpu_data_parallel
        ;;
    multi-gpu-ddp)
        setting=$multi_gpu_dist_data_parallel
        ;;
    *)
    echo Unknown setting
    exit 1
    ;;
esac

file="tasks/FINAL_TASK/pretrain.py"

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
                                                                                                                                                                                                                                                                                          