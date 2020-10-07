eval_iters=""
for (( COUNTER=6800; COUNTER<=60000; COUNTER+=200 )); do  # 10 to 40010 gap 500
    eval_iters="$eval_iters $COUNTER"
done

# eval_iters="1 101 201 301 401 501 646 647 747 847 947"
exp_name=pretrain_ndh_r2r_masked_lm_1-in-36-viewpoint

arguments="
--img_feat_dir img_features
--img_feature_file ResNet-101-faster-rcnn-genome-worientation.pickle
--data_dir tasks/NDH/data
--model_name_or_path oscar_ndh_results/$exp_name/checkpoints/
--output_dir oscar_ndh_results/$exp_name-val
--add_r2r_data
--max_seq_length 768
--img_feature_dim 2054
--eval_only
--eval_iter $eval_iters
--per_gpu_eval_batch_size 4
--drop_out 0.3
--seed 88
--num_workers 0
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

file="tasks/Viewpoint-NDH_with_Oscar/pretrain.py"

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
