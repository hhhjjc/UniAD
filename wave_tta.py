from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from models.model_helper import ModelHelper
from utils.misc_helper import to_device
from models.reconstructions.torch_wavelets import DWT_2D, IDWT_2D

# 新增多维度置信度评估器
class MultiDimensionalConfidence:
    def __init__(self):
        self.dwt = DWT_2D(wave='haar')
        
    def compute_confidence(self, feature_rec, feature_align, reconstruction_loss):
        """计算多维度置信度"""
        confidences = {}
        
        # 1. 重建误差置信度
        conf_recon = torch.exp(-reconstruction_loss / 10.0)  # 指数衰减
        confidences['reconstruction'] = conf_recon
        
        # 2. 频域能量分布一致性置信度
        conf_freq = self._frequency_consistency_confidence(feature_rec, feature_align)
        confidences['frequency'] = conf_freq
        
        # 3. 小波系数稀疏性置信度
        conf_sparse = self._sparsity_confidence(feature_rec)
        confidences['sparsity'] = conf_sparse
        
        # 4. 频带间相关性置信度
        conf_corr = self._interband_correlation_confidence(feature_rec, feature_align)
        confidences['correlation'] = conf_corr
        
        # 综合置信度
        total_confidence = (conf_recon * 0.3 + conf_freq * 0.3 + 
                           conf_sparse * 0.2 + conf_corr * 0.2)
        
        return total_confidence, confidences
    
    def _frequency_consistency_confidence(self, feature_rec, feature_align):
        # 小波分解
        rec_dwt = self.dwt(feature_rec)
        align_dwt = self.dwt(feature_align)
        
        # 计算每个频带的能量
        energy_rec = []
        energy_align = []
        C = rec_dwt.shape[1] // 4
        
        for i in range(4):
            band_rec = rec_dwt[:, i*C:(i+1)*C]
            band_align = align_dwt[:, i*C:(i+1)*C]
            energy_rec.append((band_rec**2).mean())
            energy_align.append((band_align**2).mean())
        
        # 计算能量分布的一致性
        energy_rec = torch.stack(energy_rec)
        energy_align = torch.stack(energy_align)
        consistency = 1 - torch.abs(energy_rec - energy_align).mean()
        
        return consistency.item()
    
    def _sparsity_confidence(self, feature_rec):
        # 计算小波系数的稀疏性
        rec_dwt = self.dwt(feature_rec)
        sparsity = (rec_dwt.abs() < 0.01).float().mean()
        return sparsity.item()
    
    def _interband_correlation_confidence(self, feature_rec, feature_align):
        # 计算频带间的相关性保持程度
        rec_dwt = self.dwt(feature_rec)
        align_dwt = self.dwt(feature_align)
        
        correlations = []
        C = rec_dwt.shape[1] // 4
        
        for i in range(3):  # 比较相邻频带
            band1_rec = rec_dwt[:, i*C:(i+1)*C].flatten()
            band2_rec = rec_dwt[:, (i+1)*C:(i+2)*C].flatten()
            corr_rec = torch.corrcoef(torch.stack([band1_rec, band2_rec]))[0, 1]
            
            band1_align = align_dwt[:, i*C:(i+1)*C].flatten()
            band2_align = align_dwt[:, (i+1)*C:(i+2)*C].flatten()
            corr_align = torch.corrcoef(torch.stack([band1_align, band2_align]))[0, 1]
            
            correlations.append(1 - torch.abs(corr_rec - corr_align))
        
        return torch.stack(correlations).mean().item()

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
        
        # 新增：置信度历史记录，用于动态调整策略
        self.confidence_history = []
        self.max_history_length = 100

    def forward(self, x):
        """前向传播并应用测试时适应"""
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt(
                x, self.model, self.optimizer, 
                self.entropy_margin, self.reset_threshold, self.ema
            )
            
            # 记录置信度历史
            if 'confidences' in outputs:
                avg_conf = outputs['confidences'].mean().item()
                self.confidence_history.append(avg_conf)
                if len(self.confidence_history) > self.max_history_length:
                    self.confidence_history.pop(0)
            
            if reset_flag:
                self.reset()
            self.ema = ema

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
# 修改 forward_and_adapt 函数中的样本选择部分
def forward_and_adapt(x, model, optimizer, margin, reset_threshold, ema):
    """增强的前向传播和适应函数，保持只更新filter参数"""
    optimizer.zero_grad()
    outputs = model(x)
    feature_rec = outputs["feature_rec"]
    feature_align = outputs["feature_align"]
    
    # 计算重建损失
    losses = reconstruction_loss(feature_rec, feature_align)
    
    # 新增：多维度置信度评估
    confidence_evaluator = MultiDimensionalConfidence()
    
    # 为每个样本计算置信度
    sample_confidences = []
    confidence_details = []
    for i in range(feature_rec.shape[0]):
        conf, details = confidence_evaluator.compute_confidence(
            feature_rec[i:i+1], 
            feature_align[i:i+1], 
            losses[i]
        )
        sample_confidences.append(conf)
        confidence_details.append(details)
    
    sample_confidences = torch.tensor(sample_confidences, device=losses.device)
    
    # 基于多维度置信度的样本选择
    # 修改：选择置信度最高的前20%样本
    dynamic_threshold = torch.quantile(sample_confidences, 0.8)  # 选择前20%的样本
    reliable_mask = sample_confidences > max(dynamic_threshold, 0.6)  # 提高最低置信度要求到0.6
    
    # 确保至少有一个样本（防止批次太小时没有样本被选中）
    if reliable_mask.sum() == 0 and len(sample_confidences) > 0:
        # 如果没有样本满足条件，选择置信度最高的一个
        max_conf_idx = sample_confidences.argmax()
        reliable_mask[max_conf_idx] = True
    
    if reliable_mask.sum() > 0:
        # 使用置信度加权的损失
        reliable_losses = losses[reliable_mask]
        reliable_confidences = sample_confidences[reliable_mask]
        
        # 置信度加权损失
        weighted_loss = (reliable_losses * reliable_confidences).mean()
        
        # 第一步：计算原始梯度
        weighted_loss.backward()
        
        # 根据平均置信度调整学习强度
        avg_confidence = reliable_confidences.mean()
        lr_scale = avg_confidence.item()  # 置信度越高，学习率越大
        
        # 仅对filter参数应用缩放的学习率
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                param.grad *= lr_scale
        
        # SAM第一步：计算扰动
        optimizer.first_step(zero_grad=True)
        
        # 第二次前向传播
        outputs_second = model(x)
        feature_rec_second = outputs_second["feature_rec"]
        feature_align_second = outputs_second["feature_align"]
        losses_second = reconstruction_loss(feature_rec_second, feature_align_second)
        
        # 仍然使用第一次筛选的可靠样本
        reliable_losses_second = losses_second[reliable_mask]
        weighted_loss_second = (reliable_losses_second * reliable_confidences).mean()
        
        # 第二次反向传播
        weighted_loss_second.backward()
        
        # 再次应用置信度缩放
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                param.grad *= lr_scale
        
        # SAM第二步：更新参数（仅filter相关参数）
        optimizer.second_step(zero_grad=True)
        
        # 更新EMA
        if not np.isnan(weighted_loss_second.item()):
            ema = update_ema(ema, weighted_loss_second.item())
        
        # 记录选择的样本比例（用于调试和监控）
        selection_ratio = reliable_mask.sum().item() / len(sample_confidences)
        print(f"Selected {selection_ratio*100:.1f}% samples (target: 20%), "
              f"avg confidence: {avg_confidence:.3f}")
    else:
        # 如果没有可靠样本，跳过更新
        print("No reliable samples found, skipping update")
    
    # 添加统计信息到输出
    outputs['confidences'] = sample_confidences
    outputs['confidence_details'] = confidence_details
    outputs['selection_mask'] = reliable_mask
    outputs['selection_ratio'] = reliable_mask.sum().item() / len(sample_confidences)
    
    # 检查是否需要重置
    reset_flag = False
    if ema is not None:
        # 基于置信度调整重置阈值
        avg_conf = sample_confidences.mean().item()
        adjusted_threshold = reset_threshold * (2 - avg_conf)  # 置信度低时更容易重置
        if ema < adjusted_threshold:
            print(f"Low confidence ({avg_conf:.4f}) and low EMA ({ema:.4f}), resetting model")
            reset_flag = True
    
    return outputs, ema, reset_flag

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