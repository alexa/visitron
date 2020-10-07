
arguments="
--img_feat_dir img_features
--img_feature_file ResNet-101-faster-rcnn-genome-worientation.pickle
--data_dir tasks/NDH/data
--model_name_or_path oscar_pretrained_models/base-vg-labels/ep_107_1192087
--output_dir oscar_ndh_results/temp
--max_seq_length 448
--max_img_seq_length 128
--img_feature_dim 2054
--per_gpu_train_batch_size 1
--per_gpu_eval_batch_size 1
--learning_rate 5e-05
--weight_decay 0.05
--num_iterations 50000
--warmup_steps 0
--drop_out 0.3
--logging_steps 10
--eval_logging_steps 100
--seed 88
--num_workers 1
"
# --evaluate_during_training
# --save_steps 2000

single_gpu="CUDA_VISIBLE_DEVICES=0 python"
multi_gpu_data_parallel="python"
multi_gpu_dist_data_parallel="python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0"

case $1 in
    single-gpu)
        setting=$single_gpu
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
echo $command_to_run$"\n"
eval $command_to_run

# python \
# -m torch.distributed.launch \
# --nproc_per_node 8 \
# --nnodes 1 \
# --node_rank 0 \
# CUDA_VISIBLE_DEVICES=0 \
# python \
# tasks/NDH_with_Oscar/train.py \
# --img_feat_dir img_features \
# --img_feature_file ResNet-101-faster-rcnn-genome-worientation.pickle \
# --data_dir tasks/NDH/data \
# --model_name_or_path oscar_pretrained_models/base-vg-labels/ep_107_1192087 \
# --output_dir oscar_ndh_results/temp \
# --max_seq_length 896 \
# --max_img_seq_length 256 \
# --img_feature_dim 2054 \
# --evaluate_during_training \
# --per_gpu_train_batch_size 16 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 5e-05 \
# --weight_decay 0.05 \
# --num_train_epochs 50 \
# --warmup_steps 0 \
# --drop_out 0.3 \
# --logging_steps 100 \
# --eval_logging_steps 100 \
# --save_epoch 1 \
# --seed 88 \
# --num_workers 4 \
