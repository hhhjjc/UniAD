export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
torchrun --nproc_per_node=$1 ../../tools/test_wave_tta.py --config ./config.yaml --load_path ./checkpoints/ckpt_best.pth.tar