python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 tasks/RMM/src/train.py --entity=speaker --train_datasets=CVDN --eval_datasets=CVDN --batch_size 50 --n_iters 10000
