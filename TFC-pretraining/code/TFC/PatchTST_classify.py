__all__ = ['PatchTST', 'PatchTST_Classification']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp, positional_encoding

# from models.layers.PatchTST_backbone import PatchTST_backbone
# from models.layers.PatchTST_layers import series_decomp, positional_encoding



class PatchTST_Classification(nn.Module):
    """
    patchtst分类
    """
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type:str='flatten', 
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        self.num_classes = getattr(configs, 'num_classes', 5)  
        self.c_in = getattr(configs, 'enc_in', 3)  
        context_window = getattr(configs, 'seq_len', 50)  
        
        n_layers = getattr(configs, 'e_layers', 1)
        n_heads = getattr(configs, 'n_heads', 8)
        self.d_model = getattr(configs, 'd_model', 128)  # Store d_model as an instance variable
        d_ff = getattr(configs, 'd_ff', 256)
        dropout = getattr(configs, 'dropout', 0.1)
        fc_dropout = getattr(configs, 'fc_dropout', 0.5)
        head_dropout = getattr(configs, 'head_dropout', 0.1)
        
        individual = getattr(configs, 'individual', False)
    
        patch_len = getattr(configs, 'patch_len', 6)
        stride = getattr(configs, 'stride', 3)
        padding_patch = getattr(configs, 'padding_patch', 'end')
        
        
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
        
        # print(f"000输入x shape: {x.shape}")

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

        ### wyh 预训练时候不需要分类头
        return features



if __name__ == "__main__":
    import argparse
    import platform
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST_Classification', help='net type')
    parser.add_argument('--gpu', type=int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=150, help='total training epoches')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--gamma', type=float, default=0.5, help='the gamma of focal loss')
    parser.add_argument('--beta', type=float, default=0.99, help='the beta of class balanced loss')
    parser.add_argument('--weight_d', type=float, default=0.001, help='weight decay for regularization')
    parser.add_argument('--reg_type', type=str, default='L2', choices=['L1', 'L2', 'none'], help='regularization type: L1, L2 or none')
    parser.add_argument('--save_path', type=str, default='experiments/default_run',
                        help='path for saving all outputs (checkpoints, logs, etc)') 
    parser.add_argument('--data_path', type=str,
                        default='./data/new_goat_25hz_3axis.pt', 
                        help='saved path of input data')
    parser.add_argument('--patience', type=int, default=25, help='patience for early stopping')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes in the dataset')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm for gradient clipping')
    parser.add_argument('--num_workers', type=int, default=8 if platform.system() != "Windows" else 0,
                        help='number of workers for data loading')
    parser.add_argument('--lr_decay', type=str, default='cyclic',
                        choices=['cosine', 'step', 'cyclic', 'onecycle', 'plateau'],
                        help='learning rate decay type: cosine/step/cyclic/onecycle/plateau (default: cosine)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'lion'],
                        help='optimizer to use: adamw/adam/sgd/lion (default: adamw)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--cycle_epochs', type=int, default=10, help='number of epochs per cycle for cyclic lr')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',help='device to use')
    parser.add_argument('--use_mixed_loss', type=int, default=1, help='loss type: 0 for CB Loss only, 1 for CB-CE mixed loss')
    parser.add_argument('--ce_weight', type=float, default=0.2, help='weight of CE Loss in CB-CE mixed loss')
    parser.add_argument('--weight_smooth', type=int, default=1, help='whether to apply weight smoothing: 0 for no smoothing, 1 for log smoothing')
    parser.add_argument('--focal_alpha', type=float, default=None, help='alpha parameter for focal loss, if None, use class weights')
    # T-SNE可视化相关参数
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='T-SNE困惑度参数，影响局部结构保留程度')
    parser.add_argument('--tsne_n_iter', type=int, default=1000, help='T-SNE迭代次数，影响结果质量和运行时间')
    parser.add_argument('--tsne_learning_rate', type=float, default=200.0, help='T-SNE学习率，影响收敛速度')
    parser.add_argument('--visualize_features', type=int, default=0, help='是否进行特征可视化: 0为否，1为是')

    args = parser.parse_args()


    
    model = PatchTST_Classification(configs=args)
    # Create a random tensor with shape [Batch, Input length, Channel]
    random_input = torch.randn(8, 3,50)  # Example: Batch size of 8
    output = model(random_input)
    print(output.shape)
    # print("Output:", output)