eval_iters=""
for (( COUNTER=47010; COUNTER<=50000; COUNTER+=500 )); do  # 10 to 40010 gap 500
    eval_iters="$eval_iters $COUNTER"
done

exp_name=ndh_action_selection_sample

arguments="
--img_feat_dir img_features
--img_feature_file ResNet-101-faster-rcnn-genome-worientation.pickle
--data_dir tasks/NDH/data
--model_name_or_path oscar_ndh_results/$exp_name/checkpoints/
--output_dir oscar_ndh_results/$exp_name-val
--feedback_method sample
--path_type planner_path
--max_seq_length 128
--max_img_seq_length 10
--img_feature_dim 2054
--eval_only
--eval_iter $eval_iters
--per_gpu_eval_batch_size 2
--drop_out 0.3
--seed 88
--num_workers 8
"

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
