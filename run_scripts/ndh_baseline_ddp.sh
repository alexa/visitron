python \
-m torch.distributed.launch \
--nproc_per_node 8 \
--nnodes 2 \
--node_rank 0 \
--master_addr "<REPLACE>" \
--master_port 1234 \
tasks/NDH/train.py \
--path_type=trusted_path \
--history=all \
--feedback=sample \
--eval_type=val \
--img_features ResNet-152-imagenet-pytorch \
--model_prefix DDP \
--n_iters 5000
