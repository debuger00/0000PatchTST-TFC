__all__ = ['PatchTST_Classification']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp, positional_encoding



class Configs:
    def __init__(self):
        # 基础参数
        self.enc_in = 3  # 输入特征维度
        self.seq_len = 50  # 输入序列长度
        self.pred_len = 0  # 分类任务不需要预测长度
        
        # 模型结构参数
        self.e_layers = 1  # encoder层数
        self.n_heads = 8  # 注意力头数
        self.d_model = 128  # 模型维度
        self.d_ff = 256  # 前馈网络维度
        self.dropout = 0.1  # dropout率
        self.fc_dropout = 0.5  # 全连接层dropout率
        self.head_dropout = 0.1  # 输出头dropout率
        
        # Patch相关参数
        self.patch_len = 6  # patch长度
        self.stride = 3  # patch步长
        self.padding_patch = 'end'  # patch填充方式
        
        # 其他参数
        self.individual = False  # 是否独立处理每个特征
        self.num_classes = 5  # 分类类别数


class PatchTST_Classification(nn.Module):
    """
    patchtst分类
    """
    def __init__(self, args=None, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type:str='flatten', 
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        if args is None:
            configs = Configs()
        else:
            configs = args
            
        self.num_classes = configs.num_classes
        self.c_in = configs.enc_in
        context_window = configs.seq_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        self.d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        # 计算patch数量
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':
            patch_num += 1
            
        
        self.patch_backbone = PatchTST_backbone(
            c_in=self.c_in, context_window=context_window, target_window=0,  
            patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, n_layers=n_layers, d_model=self.d_model,
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
            dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
            pretrain_head=False, head_type="flatten", individual=individual, revin=False, affine=False,
            subtract_last=False, verbose=verbose, **kwargs
        )

        self.final_patch_dim = patch_num * self.d_model
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,  # 每个通道的特征维度
            num_heads=4,        # 注意力头数
            dropout=0.1,
            batch_first=False   # 默认格式为 [seq_len, batch, embed_dim]
        )
        
        # 特征重组卷积层 
        self.feature_conv = nn.Sequential(
            # 通道分组卷积 - 每个通道独立处理
            nn.Conv1d(
                in_channels=self.c_in,
                out_channels=self.c_in*2,
                kernel_size=3,
                padding=1,
                groups=self.c_in  # depthwise卷积，每个通道单独处理
            ),
            nn.GELU(),
            # 1x1卷积进行通道融合
            nn.Conv1d(
                in_channels=self.c_in*2,
                out_channels=self.c_in,
                kernel_size=1
            ),
            nn.LayerNorm([self.c_in, self.d_model])
        )
        
        # 分类头 
        self.classifier = nn.Sequential(
            nn.Linear(self.c_in * self.d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )


    def forward(self, x):
        # 输入维度处理
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1, 2)
        elif x.dim() == 3 and x.shape[1] != self.c_in:
            x = x.transpose(1, 2)
        
        # 第一阶段特征提取
        if self.patch_backbone.padding_patch == 'end':
            x = self.patch_backbone.padding_patch_layer(x)
            
        x = x.unfold(-1, self.patch_backbone.patch_len, self.patch_backbone.stride)
        x = x.permute(0, 1, 3, 2)  # [bs, nvars, patch_len, patch_num]
        patch_features = self.patch_backbone.backbone(x)  # [bs, nvars, d_model, patch_num]
        
        bs = patch_features.shape[0]
        features = patch_features.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, d_model]
        
        # 使用全局平均池化
        features = torch.mean(features, dim=2)  # [bs, nvars, d_model]
        
        # 应用自注意力
        features_t = features.transpose(0, 1)  # [nvars, bs, d_model]
        attn_out, _ = self.attention(features_t, features_t, features_t)
        
        # 添加残差连接并转回原始形状
        features = features + attn_out.transpose(0, 1)  # [bs, nvars, d_model]
        
        # 应用卷积特征提取
        features = self.feature_conv(features)  # [bs, nvars, d_model]
        
        # 展平特征用于分类
        features = features.reshape(bs, -1)  # [bs, nvars * d_model]
        
        # # 分类头
        # output = self.classifier(features)
        # return output
        return features
    


def patchtst_classification():
    configs = Configs()
    return PatchTST_Classification(configs)