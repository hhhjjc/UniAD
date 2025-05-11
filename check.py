import torch
ckpt = torch.load("/home/huangjiacheng/work_2/UniAD/experiments/MVTec-AD_0/checkpoints/ckpt_best.pth.tar", map_location="cpu")
print(ckpt.keys())
print("Saved epoch:", ckpt.get("epoch", "Not found"))