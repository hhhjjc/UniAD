# export PYTHONPATH=../../:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=$2
# python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_val.py


# export PYTHONPATH=../../:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=$2
# python -m torch.distributed.launch --nproc_per_node=$1 --use_env ../../tools/train_val.py


export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
torchrun --nproc_per_node=$1 ../../tools/train_val.py