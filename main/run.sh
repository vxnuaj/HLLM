python3 preprocess.py

torchrun --standalone --nproc-per-node=2 --nnodes=1 python3 train.py