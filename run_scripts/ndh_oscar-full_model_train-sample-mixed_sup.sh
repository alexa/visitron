
arguments="
--img_feat_dir img_features
--img_feature_file ResNet-101-faster-rcnn-genome-worientation.pickle
--data_dir tasks/NDH/data
--model_name_or_path oscar_pretrained_models/base-vg-labels/ep_107_1192087
--output_dir oscar_ndh_results/all-layers-train_sample_mixed-sup_current-img-only_batch_2
--feedback_method sample
--path_type trusted_path
--detach_loss
--max_seq_length 128
--max_img_seq_length 10
--img_feature_dim 2054
--per_gpu_train_batch_size 2
--per_gpu_eval_batch_size 2
--learning_rate 5e-05
--weight_decay 0.05
--num_iterations 50000
--warmup_steps 0
--drop_out 0.3
--logging_steps 10
--eval_logging_steps 1000
--save_steps 500
--seed 88
--num_workers 8
"
# --evaluate_during_training

single_gpu="python"
multi_gpu_data_parallel="python"
multi_gpu_dist_data_parallel="python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0"

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

file="tasks/NDH_with_Oscar/train.py"

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
