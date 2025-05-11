import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Tuple
from .torch_wavelets import DWT_2D, IDWT_2D


# 新增类定义
class FrequencyBandAttention(nn.Module):
    """频域注意力模块，自适应关注不同频带"""
    def __init__(self, embed_dim):
        super().__init__()
        self.band_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, 4, 1),  # 4个频带
            nn.Sigmoid()
        )
        
    def forward(self, x_dwt):
        B, C, H, W = x_dwt.shape
        C_band = C // 4
        
        # 计算每个频带的重要性权重
        weights = self.band_weights(x_dwt)  # [B, 4, 1, 1]
        
        # 对每个频带应用权重
        weighted_bands = []
        for i in range(4):
            band = x_dwt[:, i*C_band:(i+1)*C_band]
            weight = weights[:, i:i+1]
            weighted_bands.append(band * weight)
        
        return torch.cat(weighted_bands, dim=1)

class ConditionalFrequencyFilter(nn.Module):
    """条件化频域滤波器"""
    def __init__(self, embed_dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 条件生成网络
        self.condition_net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, embed_dim*2, 1)  # 生成weight和bias
        )
        
        # 基础滤波器
        self.base_filter = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 生成条件参数
        condition = self.global_pool(x)
        params = self.condition_net(condition)
        weight, bias = params.chunk(2, dim=1)
        
        # 条件化调制
        out = self.base_filter(x)
        out = out * (1 + weight) + bias
        out = self.norm(out)
        out = self.activation(out)
        
        return out

class WaveMultiheadAttention(nn.Module):
    """
    基于小波变换的多头注意力机制
    - 与nn.MultiheadAttention保持相同接口
    - query在空间域，key和value在频域
    - 通过小波变换实现无损下采样
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, sr_ratio=2, device=None, dtype=None):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.sr_ratio = sr_ratio
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # 小波变换组件
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        
        # 减少特征维度的模块
        self.reduce = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.ReLU(inplace=True),
        )
        
        # # 小波域中的特征过滤器
        # self.filter = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, groups=1),
        #     nn.BatchNorm2d(embed_dim),
        #     nn.ReLU(inplace=True),
        # )
        # key和value共享同一个条件化频域滤波器
        self.shared_filter = ConditionalFrequencyFilter(embed_dim)
        # 共享的频域注意力模块
        self.freq_attention = FrequencyBandAttention(embed_dim)
        
        
        # 空间下采样（当sr_ratio > 1时）
        self.kv_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        
        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影，融合空间域和频域信息
        self.out_proj = nn.Linear(embed_dim+embed_dim//4, embed_dim)
        
        # 归一化层
        self.kv_norm = nn.LayerNorm(embed_dim)
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True, is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        保持与MultiheadAttention相同的前向传播接口
        """
        
        # 处理batch_first参数
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # 转换为 [seq_len, batch_size, embed_dim]
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        # 获取维度信息
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # 估计特征图大小（假设为方形）
        query_H = query_W = int(math.sqrt(tgt_len))
        key_H = key_W = int(math.sqrt(src_len))
        
        # 1. 处理query - 保持在空间域
        q = self.q_proj(query)
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3)  # [bsz, num_heads, tgt_len, head_dim]
        
        # 2. 处理key和value - 转换到频域
        # 重塑为图像格式进行小波变换
        k = key.transpose(0, 1)  # [bsz, src_len, embed_dim]
        v = value.transpose(0, 1)  # [bsz, src_len, embed_dim]
        
        k_img = k.view(bsz, key_H, key_W, embed_dim).permute(0, 3, 1, 2)
        v_img = v.view(bsz, key_H, key_W, embed_dim).permute(0, 3, 1, 2)
        
        # # 应用小波变换
        # k_reduced = self.reduce(k_img)
        # k_dwt = self.dwt(k_reduced)
        # k_dwt = self.filter(k_dwt)
        
        # v_reduced = self.reduce(v_img)
        # v_dwt = self.dwt(v_reduced)
        # v_dwt = self.filter(v_dwt)
        def process_kv(kv_img):
            kv_reduced = self.reduce(kv_img)
            kv_dwt = self.dwt(kv_reduced)
            kv_dwt = self.freq_attention(kv_dwt)
            kv_dwt = self.shared_filter(kv_dwt)
            return kv_dwt
        
        # 应用相同的处理流程
        k_dwt = process_kv(k_img)
        v_dwt = process_kv(v_img)
        
        # 应用空间降采样并投影
        k_embed = self.kv_embed(k_dwt).reshape(bsz, embed_dim, -1).permute(0, 2, 1)
        v_embed = self.kv_embed(v_dwt).reshape(bsz, embed_dim, -1).permute(0, 2, 1)
        
        # 应用归一化
        k_embed = self.kv_norm(k_embed)
        v_embed = self.kv_norm(v_embed)
        
        # 投影到多头格式
        k_proj = self.k_proj(k_embed).view(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_proj = self.v_proj(v_embed).view(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 3. 计算注意力分数和加权值
        attn_weights = (q @ k_proj.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # 应用注意力掩码（如果有）
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_weights += attn_mask
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
                attn_weights += attn_mask
        
        # 应用键填充掩码（如果有）
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # 软最大化注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 计算注意力输出
        attn_output = (attn_weights @ v_proj).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        
        # 4. 应用逆小波变换生成增强特征
        v_idwt = self.idwt(v_dwt)
        v_idwt = v_idwt.view(bsz, -1, query_H * query_W).transpose(1, 2)
        
        # 5. 合并注意力输出和重建特征
        combined_output = torch.cat([attn_output, v_idwt], dim=-1)
        output = self.out_proj(combined_output).transpose(0, 1)  # [tgt_len, bsz, embed_dim]
        
        # 处理输出格式
        if self.batch_first and is_batched:
            output = output.transpose(1, 0)
        
        # 处理注意力权重的输出
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            
            if self.batch_first and is_batched:
                return output, attn_weights
            else:
                return output, attn_weights
        else:
            return output, None