# export PYTHONPATH=../../:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=$2
# python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_val.py


# export PYTHONPATH=../../:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=$2
# python -m torch.distributed.launch --nproc_per_node=$1 --use_env ../../tools/train_val.py


# export PYTHONPATH=../../:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=$2
# torchrun --nproc_per_node=$1 ../../tools/train_val.py

#!/bin/bash
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=cdtrans_recon \
python -u ../../tools/train_cdtrans_recon.py