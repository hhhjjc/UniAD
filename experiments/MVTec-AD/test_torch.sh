# 使用TTA进行测试
export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
torchrun --nproc_per_node=$1 ../../tools/test_tta.py \
    --checkpoint checkpoints/ckpt.pth.tar \
    --tta_steps 3 \
    --tta_lr 1e-5 \
    --tta_threshold 0.5