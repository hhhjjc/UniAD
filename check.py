import torch
ckpt = torch.load("/home/huangjiacheng/work_2/UniAD/experiments/MVTec-AD/checkpoints/ckpt.pth.tar", map_location="cpu")
print(ckpt.keys())
print("Saved epoch:", ckpt.get("epoch", "Not found"))