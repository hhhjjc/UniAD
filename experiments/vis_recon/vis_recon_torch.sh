# class_name: bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
python -u ../../tools/vis_recon.py --class_name $1
