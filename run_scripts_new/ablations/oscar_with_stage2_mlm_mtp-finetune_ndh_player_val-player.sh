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

file="tasks/FINAL_TASK/train.py"

# eval_iters="1 101 201 301 401 501 601 646 647 747 847 947 1047 1147 1247 1292 1293 1393 1493 1593 1693 1793 1893 1938 1939 2039 2139 2239 2339 2439 2539 2584 2585 2685 2785 2885 2985 3085 3185 3230 3231 3331 3431 3531 3631 3731 3831 3876 3877 3977 4077 4177 4277 4377 4477 4522 4523 4623 4723 4823 4923 5023 5123 5168 5169 5269 5369 5469 5569 5669 5769 5814 5815 5915 6015 6115 6215 6315 6415 6460 6461 6561 6661 6761 6861 6961 7061 7106 7107 7207 7307 7407 7507 7607 7707 7752 7753 7853 7953 8053 8153 8253 8353 8398 8399 8499 8599 8699 8799 8899 8999 9044 9045 9145 9245 9345 9445 9545 9645 9690 9691 9791 9891 9991 10091 10191 10291 10336 10337 10437 10537 10637 10737 10837 10937 10982 10983 11083 11183 11283 11383 11483 11583 11628 11629 11729 11829 11929 12029 12129 12229 12274 12275 12375 12475 12575 12675 12775 12875 12920"
eval_iters=""
for (( COUNTER=10; COUNTER<=50000; COUNTER+=100 )); do  # 10 to 40010 gap 500
    eval_iters="$eval_iters $COUNTER"
done

# eval_iters="$eval_iters 10000"

exp_name=srv_2/results/viewpoint-2/ndh-oscar_stage2_lm_mtp-player

arguments="
--img_feat_dir srv/img_features
--img_feature_file ResNet-152-imagenet.tsv
--data_dir srv/task_data/NDH/data
--model_name_or_path $exp_name/checkpoints/
--output_dir $exp_name-val-player
--path_type player_path
--max_seq_length 768
--img_feature_dim 2054
--lstm_img_feature_dim 2048
--eval_only
--eval_iter $eval_iters
--per_gpu_eval_batch_size 4
--drop_out 0.3
--seed 88
--num_workers 0
--slurm_info '$slurm_info'
"

command_to_run="${setting} ${file} ${arguments}"
echo $command_to_run
echo
eval $command_to_run
