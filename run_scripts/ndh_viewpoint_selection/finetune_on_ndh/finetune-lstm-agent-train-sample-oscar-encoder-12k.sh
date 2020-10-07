
arguments="
--img_feat_dir img_features
--img_feature_file ResNet-152-imagenet-pytorch.tsv
--data_dir tasks/NDH/data
--model_name_or_path oscar_ndh_results/pretrain_masked_lm_1-in-36-viewpoint/checkpoints/checkpoint-12920/
--output_dir oscar_ndh_results/finetune_viewpoint-selection_lstm-agent_planner-path_sample-oscar-encoder-12k
--agent att-lstm
--encoder oscar
--feedback_method sample
--path_type planner_path
--max_seq_length 768
--img_feature_dim 2054
--lstm_img_feature_dim 2048
--per_gpu_train_batch_size 8
--per_gpu_eval_batch_size 16
--learning_rate 5e-05
--weight_decay 0.05
--num_iterations 50000
--warmup_steps 0
--drop_out 0.3
--logging_steps 10
--eval_logging_steps 500
--save_steps 100
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

file="tasks/Viewpoint-NDH_with_Oscar/train.py"

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
