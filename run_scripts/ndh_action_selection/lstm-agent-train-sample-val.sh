eval_iters=""
for (( COUNTER=10; COUNTER<=50000; COUNTER+=100 )); do  # 10 to 40010 gap 500
    eval_iters="$eval_iters $COUNTER"
done
exp_name=action_prediction_lstm-agent-try-2


arguments="
--img_feat_dir img_features
--img_feature_file ResNet-152-imagenet-pytorch.tsv
--data_dir tasks/NDH/data
--model_name_or_path oscar_ndh_results/$exp_name/checkpoints/
--output_dir oscar_ndh_results/$exp_name-val
--agent att-lstm
--feedback_method sample
--path_type planner_path
--img_feature_dim 2054
--lstm_img_feature_dim 2048
--aemb 32
--eval_only
--eval_iter $eval_iters
--per_gpu_eval_batch_size 16
--drop_out 0.3
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

file="tasks/Action-NDH_with_Oscar/train.py"

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
