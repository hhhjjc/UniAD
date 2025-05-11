# import torch, torchinfo 
# ckpt = torch.load("/home/huangjiacheng/work_2/UniAD/experiments/AD_1/checkpoints/ckpt_best.pth.tar", map_location="cpu")
# # print(ckpt.keys())
# # print("Saved epoch:", ckpt.get("epoch", "Not found"))
# torchinfo.summary(ckpt, (1, 3, 224, 224))

import torch
import sys
import os

def view_checkpoint(checkpoint_path):
    """查看checkpoint的内容"""
    if not os.path.exists(checkpoint_path):
        print(f"文件不存在: {checkpoint_path}")
        return
    
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 显示checkpoint的顶层keys
    print("\n=== Checkpoint Keys ===")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    # 显示epoch和best_metric（如果有）
    if 'epoch' in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    if 'best_metric' in checkpoint:
        print(f"Best metric: {checkpoint['best_metric']}")
    
    # 显示state_dict的参数
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\n=== State Dict Parameters ({len(state_dict)} total) ===")
        
        # 按模块分组显示
        modules = {}
        for name, param in state_dict.items():
            module_name = name.split('.')[0]
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append((name, param.shape))
        
        # 显示每个模块的参数
        for module_name, params in modules.items():
            print(f"\n[{module_name}] ({len(params)} parameters)")
            for name, shape in params[:5]:  # 只显示前5个
                print(f"  {name}: {shape}")
            if len(params) > 5:
                print(f"  ... and {len(params)-5} more")

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("用法: python view_checkpoint.py <checkpoint_path>")
    #     sys.exit(1)
    
    checkpoint_path = 'experiments/AD_1/checkpoints/ckpt_best.pth.tar'
    view_checkpoint(checkpoint_path)