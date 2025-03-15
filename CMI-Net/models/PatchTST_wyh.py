__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from models.layers.PatchTST_backbone import PatchTST_backbone
from models.layers.PatchTST_layers import series_decomp

# from layers.PatchTST_backbone import PatchTST_backbone
# from layers.PatchTST_layers import series_decomp

class Configs:
    def __init__(self):
        # 基础参数
        self.enc_in = 3         # 输入特征维度
        self.seq_len = 50        # 输入序列长度
        self.pred_len = 24       # 预测序列长度
        
        # 模型结构参数
        self.e_layers = 2        # encoder层数
        self.n_heads = 8         # 注意力头数
        self.d_model = 128       # 模型维度
        self.d_ff = 256         # 前馈网络维度
        self.dropout = 0.2       # dropout率
        self.fc_dropout = 0.2    # 全连接层dropout率
        self.head_dropout = 0.2  # 输出头dropout率
        
        # Patch相关参数 
        self.patch_len = 5      # patch长度
        self.stride = 3          # patch步长
        self.padding_patch = 'end'  # patch填充方式
        
        # 1D卷积参数
        self.conv1d_kernel_size = 3  # 1D卷积核大小
        self.conv1d_out_channels = 32  # 1D卷积输出通道数
        
        # 数据处理参数
        self.individual = False   # 是否独立处理每个特征
        self.revin = False        # 是否使用RevIN
        self.affine = False       # RevIN是否使用affine变换
        self.subtract_last = False  # 是否减去最后一个值
        
        # 分解相关参数
        self.decomposition = False  # 是否使用分解
        self.kernel_size = 25      # 分解核大小

     # 修改预测相关参数为分类参数
        self.num_classes = 5     # 分类类别数
        
        
        # 分类器特定参数
        self.classifier_dropout = 0.2  # 分类器dropout率
        self.use_weighted_loss = False # 是否使用加权损失（处理类别不平衡）

        # 1D卷积参数
        self.conv1d_kernel_size = 3  # 1D卷积核大小
        self.conv1d_out_channels = 32  # 1D卷积输出通道数

        
        # 计算特征维度 - 确保结果是整数
        # 公式1: patch个数 = 取整（（输入序列长度-patch长度）/步长）+ 2
        patch_num = int((self.seq_len - self.patch_len) / self.stride) + 2
        # 公式2: 特征维度 = patch个数 * 模型维度
        self.feature_dim = int(patch_num * self.d_model)  # 特征维


class PatchTSTNet(nn.Module):
    def __init__(self, configs, 
                 max_seq_len:Optional[int]=1024,     # 最大序列长度，默认1024
                 d_k:Optional[int]=None,             # 注意力机制中key的维度
                 d_v:Optional[int]=None,             # 注意力机制中value的维度  
                 norm:str='BatchNorm',               # 归一化方法，默认使用BatchNorm
                 attn_dropout:float=0.1,              # 注意力层的dropout率
                 act:str="gelu",                     # 激活函数，默认使用GELU
                 key_padding_mask:bool='auto',       # 是否使用key padding mask
                 padding_var:Optional[int]=None,     # padding的值
                 attn_mask:Optional[Tensor]=None,    # 注意力mask
                 res_attention:bool=True,            # 是否使用残差注意力连接
                 pre_norm:bool=False,                # 是否在注意力之前使用归一化
                 store_attn:bool=False,              # 是否存储注意力权重
                 pe:str='zeros',                     # 位置编码类型
                 learn_pe:bool=True,                 # 是否学习位置编码
                 pretrain_head:bool=False,           # 是否使用预训练头
                 head_type = 'flatten',              # 输出头类型
                 verbose:bool=False,                 # 是否打印详细信息
                 **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        feature_dim = configs.feature_dim

        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=configs.conv1d_out_channels, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=configs.conv1d_out_channels, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=configs.conv1d_out_channels, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        

        
        # 修改分类头，增加特征提取能力
        self.classifier = nn.Sequential(

            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Dropout(configs.classifier_dropout),
            
            # 最终分类层
            nn.Linear(feature_dim//2, configs.num_classes)
        )
        
    
    def forward(self, x, return_probs=True):           # x: [Batch, Input length, Channel]
        # 1. 去掉第二维的1，转换为 [batch_size, 50, 3]
        x = x.squeeze(1)
       
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)   # [Batch, conv_out_channels, num_patches]
            x = x.permute(0,2,1)
        
        x = x.mean(dim=-1)
        # 主分类器的logits
        logits = self.classifier(x)  # [Batch, num_classes]
        
        # 根据需要返回概率值或logits
        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits

def PatchTST():
    configs = Configs()
    return PatchTSTNet(configs)

def PatchTST():
    configs = Configs()
    return PatchTSTNet(configs)

# 使用示例
if __name__ == "__main__":
    configs = Configs()
    model = PatchTSTNet(configs)
    
    # 测试模型并打印每一步的形状
    batch_size = 32
    # 创建新的输入形状 [batch_size, 1, 50, 3]
    x = torch.randn(batch_size, 1, configs.seq_len, configs.enc_in)
    print(f"原始输入形状: {x.shape}")  # 预期: [32, 1, 50, 3]
    
    # 测试数据流经模型的形状变化
    with torch.no_grad():
        # 1. 去掉第二维的1，转换为 [batch_size, 50, 3]
        x = x.squeeze(1)
        print(f"去除维度1后形状: {x.shape}")  # 预期: [32, 50, 3]
        
        # 2. 转置为 [Batch, Channel, Length]
        x_permuted = x.permute(0, 2, 1)
        print(f"第一次转置后形状: {x_permuted.shape}")  # 预期: [32, 3, 50]
        
        # 3. 通过backbone
        if not model.decomposition:
            backbone_output = model.model(x_permuted)
            print(f"Backbone输出形状: {backbone_output.shape}")
            
            # 4. 转置回 [Batch, Length, Channel]
            output_permuted = backbone_output.permute(0, 2, 1)
            print(f"第二次转置后形状: {output_permuted.shape}")
            
            # 5. 计算序列维度的平均值
            x = output_permuted.mean(dim= -1)  # [Batch, Channel]
            print(f"平均池化后形状: {x.shape}")
            
            # 6. 最终分类输出
            final_output = model.classifier(x)
            print(f"分类器输出形状: {final_output.shape}")  # 预期: [32, num_classes]
            
            # 7. 测试概率输出
            probs = F.softmax(final_output, dim=-1)
            print(f"概率输出形状: {probs.shape}")  # 预期: [32, num_classes]