import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from models.reconstructions.uniad import UniAD, build_position_embedding
from models.initializer import initialize_from_cfg

class CDTransRecon(UniAD):
    """Cross-Domain Transformer for anomaly detection on new categories"""
    
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        num_prototypes=15,  # 类别原型数量
        **kwargs,
    ):
        super().__init__(
            inplanes, 
            instrides, 
            feature_size, 
            feature_jitter, 
            neighbor_mask, 
            hidden_dim, 
            pos_embed_type, 
            save_recon, 
            initializer, 
            **kwargs
        )
        
        # 类别原型库
        self.num_prototypes = num_prototypes
        self.prototype_embed = nn.Embedding(num_prototypes, hidden_dim)
        
        # 替换原始Transformer为跨域Transformer
        self.transformer = CrossDomainTransformer(
            hidden_dim, feature_size, neighbor_mask, num_prototypes=num_prototypes, **kwargs
        )
        
        # 自适应层
        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        initialize_from_cfg(self, initializer)
    
    def forward(self, input, prototype_idx=None):
        # 基本特征提取与UniAD相同
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(feature_align, "b c h w -> (h w) b c")  # (H x W) x B x C
        
        # 特征抖动（训练时）
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        
        # 投影到隐藏维度
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        
        # 选择类别原型（如果未指定则使用相似度匹配）
        if prototype_idx is None and not self.training:
            prototype_idx = self._match_prototypes(feature_tokens)
        elif prototype_idx is None and self.training:
            # 训练时随机选择原型
            batch_size = feature_tokens.size(1)
            prototype_idx = torch.randint(0, self.num_prototypes, (batch_size,)).to(feature_tokens.device)
        
        # 获取选定的原型
        selected_prototypes = self.prototype_embed(prototype_idx)  # B x C
        
        # 执行跨域重建
        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed, selected_prototypes
        )  # (H x W) x B x C
        
        # 后处理与UniAD相同
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W
        
        # 保存重建结果（测试时）
        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)
            # self._save_reconstructions(input, feature_rec)
        
        # 计算异常分数
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
            "prototype_idx": prototype_idx
        }
    
    def _match_prototypes(self, feature_tokens):
        """匹配最相似的类别原型"""
        # 计算全局特征表示
        global_feat = feature_tokens.mean(dim=0)  # B x C
        
        # 计算与所有原型的相似度
        prototypes = self.prototype_embed.weight  # num_prototypes x C
        similarity = torch.matmul(global_feat, prototypes.t())  # B x num_prototypes
        
        # 选择最相似的原型
        matched_idx = similarity.argmax(dim=1)  # B
        return matched_idx
    
    def update_prototypes(self, features_dict):
        """更新类别原型库（可选的测试时自适应）"""
        with torch.no_grad():
            for cls_idx, features in features_dict.items():
                if cls_idx >= self.num_prototypes:
                    continue
                # 计算平均特征
                avg_feature = torch.mean(features, dim=0)
                # 更新原型（使用指数移动平均）
                alpha = 0.9  # 动量因子
                current_proto = self.prototype_embed.weight[cls_idx]
                updated_proto = alpha * current_proto + (1 - alpha) * avg_feature
                self.prototype_embed.weight[cls_idx] = F.normalize(updated_proto, dim=0)


class CrossDomainTransformer(nn.Module):
    """实现跨域知识迁移的Transformer"""
    
    def __init__(
        self,
        hidden_dim,
        feature_size,
        neighbor_mask,
        num_prototypes,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask
        self.num_prototypes = num_prototypes
        
        # 编码器
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        
        # 跨域解码器
        cross_decoder_layer = CrossDomainDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.cross_decoder = TransformerDecoder(
            cross_decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        
        self.hidden_dim = hidden_dim
        self.nhead = nhead
    
    def generate_mask(self, feature_size, neighbor_size):
        """生成注意力掩码矩阵"""
        # 与UniAD相同的掩码生成
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask
    
    def forward(self, src, pos_embed, prototypes):
        """前向传播，包含编码和跨域解码"""
        _, batch_size, _ = src.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C
        
        # 生成掩码
        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None
        
        # 编码器处理
        output_encoder = self.encoder(
            src, mask=mask_enc, pos=pos_embed
        )  # (H X W) x B x C
        
        # 跨域解码器处理
        output_decoder = self.cross_decoder(
            output_encoder,
            prototypes,  # 类别原型作为跨域知识源
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C
        
        return output_decoder, output_encoder


class CrossDomainDecoderLayer(nn.Module):
    """跨域解码器层，实现源域到目标域的知识迁移"""
    
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        
        # 查询嵌入，与UniAD类似但更灵活
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C
        
        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        
        # 跨域注意力模块
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        
        # 规范化层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = self._get_activation_fn(activation)
        self.normalize_before = normalize_before
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(
        self,
        tgt,
        prototype,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
    ):
        """前向传播，包括自注意力和跨域注意力"""
        # 获取批次大小
        _, batch_size, _ = tgt.shape
        
        # 初始化查询嵌入
        query = self.learned_embed.weight
        query = query.unsqueeze(1).expand(-1, batch_size, -1)  # (H x W) x B x C
        
        # 添加位置编码
        if pos is not None:
            query = query + pos
        
        # 自注意力
        q = k = query
        src2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        query = query + self.dropout1(src2)
        query = self.norm1(query)
        
        # 跨域注意力
        # 使用类别原型作为跨域知识的键和值
        prototype_expand = prototype.unsqueeze(0).expand(query.size(0), -1, -1)  # (H x W) x B x C
        src2 = self.cross_attn(
            query=query,
            key=prototype_expand,
            value=prototype_expand,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        query = query + self.dropout2(src2)
        query = self.norm2(query)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm3(query)
        
        return query
    
    @staticmethod
    def _get_activation_fn(activation):
        """返回激活函数"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# 以下组件与UniAD相同，保持兼容性
from models.reconstructions.uniad import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer