export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
torchrun --nproc_per_node=$1 ../../tools/test_wave_tta.py --config ./config_test.yaml --load_path /home/huangjiacheng/work_2/UniAD/experiments/AD_1/checkpoints/ckpt_best.pth.tar