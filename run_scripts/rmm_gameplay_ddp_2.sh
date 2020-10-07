python \
-m torch.distributed.launch \
--master_addr "<REPLACE>" \
--master_port 1234 \
--nproc_per_node 8 \
--nnodes 4 \
--node_rank 1 \
tasks/RMM/src/train.py \
--mode=gameplay \
--eval_branching=3 \
--action_probs_branching \
--train_datasets=CVDN \
--eval_datasets=CVDN \
--batch_size 10 \
--n_iters 10000
