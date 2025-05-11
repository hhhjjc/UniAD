from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from models.model_helper import ModelHelper
from utils.misc_helper import to_device

def update_ema(ema, new_data):
    """更新指数移动平均值"""
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class WaveAD_TTA(nn.Module):
    """WaveAD模型的测试时适应实现
    专注于更新WaveAD中的filter函数，作为测试时的适应参数
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, entropy_margin=50, reset_threshold=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "TTA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # 重建损失的阈值参数
        self.entropy_margin = entropy_margin  
        self.reset_threshold = reset_threshold
        self.ema = None  # 记录移动平均损失，用于模型恢复判断

        # 保存初始模型状态，以便在需要时重置
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        """前向传播并应用测试时适应"""
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt(x, self.model, self.optimizer, 
                                                     self.entropy_margin, self.reset_threshold, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # 更新移动平均损失值

        return outputs

    def reset(self):
        """重置模型和优化器到初始状态"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                self.model_state, self.optimizer_state)
        self.ema = None

@torch.jit.script
def reconstruction_loss(feature_rec: torch.Tensor, feature_align: torch.Tensor) -> torch.Tensor:
    """计算重建损失"""
    return torch.mean((feature_rec - feature_align) ** 2, dim=1)

@torch.enable_grad()  # 确保在测试模式下也能计算梯度
def forward_and_adapt(x, model, optimizer, margin, reset_threshold, ema):
    """前向传播并适应模型到输入数据
    专注于更新WaveAD中的filter函数参数
    """
    optimizer.zero_grad()
    # 前向传播
    outputs = model(x)
    feature_rec = outputs["feature_rec"]
    feature_align = outputs["feature_align"]
    
    # 计算重建损失
    losses = reconstruction_loss(feature_rec, feature_align)
    
    # 筛选可靠样本（损失低于阈值的样本）
    reliable_idx = torch.where(losses < margin)
    if len(reliable_idx[0]) > 0:
        reliable_losses = losses[reliable_idx]
        loss = reliable_losses.mean()
        
        # 第一步：计算原始梯度
        loss.backward()
        
        # SAM第一步：计算扰动
        optimizer.first_step(zero_grad=True)
        
        # 第二次前向传播
        outputs_second = model(x)
        feature_rec_second = outputs_second["feature_rec"]
        feature_align_second = outputs_second["feature_align"]
        losses_second = reconstruction_loss(feature_rec_second, feature_align_second)
        
        # 仍然使用第一次筛选的可靠样本索引
        reliable_losses_second = losses_second[reliable_idx]
        
        # 记录第二次损失值，用于模型恢复策略
        loss_second_value = reliable_losses_second.detach().mean(0)
        
        # 再次筛选可靠样本
        reliable_idx_second = torch.where(losses_second < margin)
        if len(reliable_idx_second[0]) > 0:
            final_losses = losses_second[reliable_idx_second]
            loss_second = final_losses.mean()
            
            # 第二次反向传播
            loss_second.backward()
            
            # SAM第二步：更新参数
            optimizer.second_step(zero_grad=True)
            
            # 更新移动平均损失
            if not np.isnan(loss_second.item()):
                ema = update_ema(ema, loss_second.item())
        else:
            # 如果第二次没有可靠样本，仍然执行第二步，但不更新EMA
            optimizer.second_step(zero_grad=True)
        
        # 检查是否需要重置模型
        reset_flag = False
        if ema is not None and ema < reset_threshold:
            print(f"Loss too low (EMA: {ema:.4f}), resetting model")
            reset_flag = True
        
        return outputs, ema, reset_flag
    else:
        # 如果没有可靠样本，跳过更新
        return outputs, ema, False

def collect_filter_params(model):
    """收集WaveAD中filter函数的参数"""
    params = []
    names = []
    
    def find_filter_in_module(module, prefix=''):
        for name, m in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 找到WaveAD中的filter模块
            if name == 'filter' and isinstance(m, nn.Sequential) and hasattr(m, 'weight'):
                for param_name, param in m.named_parameters():
                    param.requires_grad_(True)
                    params.append(param)
                    names.append(f"{full_name}.{param_name}")
            
            # 特别处理WaveMultiheadAttention中的filter
            elif 'WaveMultiheadAttention' in m.__class__.__name__:
                if hasattr(m, 'filter'):
                    for param_name, param in m.filter.named_parameters():
                        param.requires_grad_(True)
                        params.append(param)
                        names.append(f"{full_name}.filter.{param_name}")
                        
            elif len(list(m.children())) > 0:
                find_filter_in_module(m, full_name)
    
    # 从reconstruction模块开始查找
    reconstruction = model.reconstruction
    find_filter_in_module(reconstruction)
    
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """复制模型和优化器状态，以便之后重置"""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """从保存的副本中恢复模型和优化器状态"""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """配置模型用于测试时适应"""
    # 训练模式，因为我们需要更新参数
    model.train()
    # 禁用所有参数的梯度计算
    model.requires_grad_(False)
    
    # 重点：只有filter函数需要梯度
    reconstruction = model.reconstruction
    
    # 递归查找并启用filter参数的梯度
    def enable_filter_grad(module):
        for name, m in module.named_children():
            if name == 'filter' and isinstance(m, nn.Sequential):
                for param in m.parameters():
                    param.requires_grad_(True)
            
            # 特别处理WaveMultiheadAttention
            elif 'WaveMultiheadAttention' in m.__class__.__name__:
                if hasattr(m, 'filter'):
                    for param in m.filter.parameters():
                        param.requires_grad_(True)
                        
            elif len(list(m.children())) > 0:
                enable_filter_grad(m)
    
    enable_filter_grad(reconstruction)
    
    return model