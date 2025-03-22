import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)


class PatchTSTConfig:
    def __init__(self):
        # 基础参数
        self.num_input_channels = 3         # 输入特征维度
        self.context_length = 50            # 输入序列长度
        self.prediction_length = 0          # 预测序列长度
        
        # Transformer结构参数
        self.num_layers = 3                 # encoder层数
        self.num_attention_heads = 8        # 注意力头数
        self.d_model = 128                  # 模型维度
        self.feedforward_dim = 256          # 前馈网络维度
        self.dropout = 0.3                  # dropout率
        self.attention_dropout = 0.2        # 注意力层的dropout率
        self.head_dropout = 0.5             # 输出头dropout率
        
        # Patch相关参数 
        self.patch_length = 12              # patch长度
        self.stride = 6                     # patch步长
        self.padding_type = 'end'           # patch填充方式
        
        # 数据处理参数
        self.use_bias = True                # 使用偏置
        self.do_mask_input = False          # 是否使用掩码
        self.use_cls_token = True           # 是否使用cls_token，分类任务常设置为True
        
        # 分类相关参数
        self.num_targets = 5                # 分类类别数
        
        # 卷积相关参数
        self.use_conv1d = True              # 是否使用1D卷积层
        self.conv1d_kernel_size = 3         # 卷积核大小
        self.conv1d_out_channels = 128      # 输出通道数
        
        # 其他参数
        self.use_return_dict = True         

class PatchTSTModelOutput:
    def __init__(
        self,
        last_hidden_state: torch.FloatTensor = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
    ):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions

class PatchTSTForClassificationOutput:
    def __init__(
        self,
        loss: Optional[torch.FloatTensor] = None,
        prediction_logits: torch.FloatTensor = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
    ):
        self.loss = loss
        self.prediction_logits = prediction_logits
        self.hidden_states = hidden_states
        self.attentions = attentions

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)

class PatchTSTPreTrainedModel(nn.Module):
    
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def post_init(self):
        self.init_weights()
    
    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        self.apply(_init_weights)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model)
        )
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, attention_mask=None):
        # 自注意力层    
        residual = x
        x = self.layer_norm1(x)
        x, attention_weights = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x, attention_weights

class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        
        # 1D卷积层      
        self.use_conv1d = config.use_conv1d
        if self.use_conv1d:
            self.conv1d = nn.Conv1d(
                in_channels=config.num_input_channels,
                out_channels=config.conv1d_out_channels,
                kernel_size=config.conv1d_kernel_size,
                padding=config.conv1d_kernel_size // 2,
                bias=config.use_bias
            )
            self.adapt_pool = nn.AdaptiveAvgPool1d(config.context_length)
            input_channels = config.conv1d_out_channels
        else:
            input_channels = config.num_input_channels
        
        # Save input_channels as instance variable
        self.input_channels = input_channels
        
        # Patch嵌入
        self.patch_length = config.patch_length
        self.stride = config.stride
        self.padding_type = config.padding_type
        
        # 计算patch数量
        seq_len = config.context_length
        patch_len = config.patch_length
        stride = config.stride
        
        # 计算patch个数
        num_patches = (seq_len - patch_len) // stride + 1
        if (seq_len - patch_len) % stride != 0:
            num_patches += 1
        self.num_patches = num_patches
        
        # 如果使用填充
        if self.padding_type == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.num_patches += 1
            
        # Patch编码
        self.patch_projection = nn.Linear(patch_len, config.d_model)
        
        # 位置编码 - 考虑CLS token
        max_len = self.num_patches + (1 if config.use_cls_token else 0)
        self.position_encoding = PositionalEncoding(config.d_model, max_len=max_len)
        self.dropout = nn.Dropout(config.dropout)
        
        # 是否使用CLS token
        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # 权重初始化
        self.post_init()
        
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 参数设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        hidden_states = []
        attention_weights = []
        
        # 去掉维度1
        x = past_values.squeeze(1) if past_values.dim() == 4 else past_values
        
        # 应用1D卷积
        if self.use_conv1d:
            # 转换维度便于卷积
            x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
            x = self.conv1d(x)
            x = F.relu(x)
            x = self.adapt_pool(x)  # 保持序列长度不变
        else:
            # 如果不使用卷积，直接转置
            x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        
        # Patching处理
        if self.padding_type == 'end':
            x = self.padding_layer(x)
            
        # 将序列划分为patches
        x = x.unfold(dimension=-1, size=self.patch_length, step=self.stride)  # [batch, channels, num_patches, patch_len]
        
        batch_size, channels, num_patches, patch_len = x.shape
        
    
        x = x.permute(0, 2, 1, 3)  # [batch, num_patches, channels, patch_len]
        x = x.reshape(batch_size * num_patches, self.input_channels, patch_len)  # [batch*num_patches, channels, patch_len]
        

        x = x.transpose(1, 2).reshape(batch_size * num_patches * channels, patch_len)  # [batch*num_patches*channels, patch_len]
        
        # 编码每个patch
        x = self.patch_projection(x)  # [batch*num_patches*channels, d_model]
        
        x = x.reshape(batch_size, num_patches, channels, -1)  # [batch, num_patches, channels, d_model]
        
        # 平均化通道维度
        x = x.mean(dim=2)  # [batch, num_patches, d_model]
        
        # 如果使用CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, 1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # [batch, num_patches+1, d_model]
        
        # 添加位置编码 - 使用position_encoding的forward方法
        x = x + self.position_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 应用Transformer编码器层
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        if output_hidden_states:
            all_hidden_states.append(x)
        
        # 顺序应用编码器层
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            
            if output_hidden_states:
                all_hidden_states.append(x)
            if output_attentions:
                all_attentions.append(attn_weights)
        
        # 最终输出
        if not return_dict:
            return tuple(v for v in [x, all_hidden_states, all_attentions] if v is not None)
        
        return PatchTSTModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

# 定义PatchTST分类头
class PatchTSTClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        
        # 计算特征维度
        if config.use_cls_token:
            # 如果使用CLS token，只使用CLS token的特征
            feature_dim = config.d_model
        else:
            # 如果不使用CLS token，使用所有token的平均特征
            feature_dim = config.d_model
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(512, config.num_targets)
        )
        
    def forward(self, hidden_states):
        # hidden_states形状: [batch, seq_len, hidden]
        
        if self.use_cls_token:
            # 如果使用CLS token，只使用第一个token (CLS token)
            x = hidden_states[:, 0]  # [batch, hidden]
        else:
            # 否则使用所有token的平均值
            x = hidden_states.mean(dim=1)  # [batch, hidden]
        
        # 应用分类器
        logits = self.classifier(x)
        
        return logits

# 定义PatchTST分类模型
class PatchTSTForClassification(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        self.model = PatchTSTModel(config)
        self.head = PatchTSTClassificationHead(config)

        # 初始化权重
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        past_observed_mask: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PatchTSTForClassificationOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if target_values is not None:
            loss = nn.CrossEntropyLoss()
            loss_val = loss(y_hat, target_values)

        if not return_dict:
            outputs = (y_hat,) + (model_output.hidden_states, model_output.attentions)
            outputs = (loss_val,) + outputs if loss_val is not None else outputs
            return outputs
            
        return PatchTSTForClassificationOutput(
            loss=loss_val,
            prediction_logits=y_hat,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )
    
    def get_features(self, x):
        """获取特征向量的方法，用于特征可视化"""
        # 前向传播获取特征
        model_output = self.model(past_values=x, return_dict=True)
        hidden_states = model_output.last_hidden_state
        
        # 提取特征
        if self.config.use_cls_token:
            # 如果使用CLS token，只使用第一个token (CLS token)
            features = hidden_states[:, 0]  # [batch, hidden]
        else:
            # 否则使用所有token的平均值
            features = hidden_states.mean(dim=1)  # [batch, hidden]
        
        # 应用分类器获取预测
        logits = self.head(model_output.last_hidden_state)
        
        # 返回特征和概率
        return features, F.softmax(logits, dim=-1)

# 创建一个实例化函数
def get_patchtst_for_classification():
    config = PatchTSTConfig()
    return PatchTSTForClassification(config)

# 测试代码
if __name__ == "__main__":
    # 创建配置和模型
    config = PatchTSTConfig()
    model = PatchTSTForClassification(config)
    
    # 打印模型结构
    print(model)
    
    # 创建测试输入
    batch_size = 32
    x = torch.randn(batch_size, 1, config.context_length, config.num_input_channels)
    
    # 测试前向传播
    output = model(past_values=x)
    
    # 打印输出形状
    if isinstance(output, PatchTSTForClassificationOutput):
        print(f"输出预测形状: {output.prediction_logits.shape}")
    else:
        print(f"输出预测形状: {output[0].shape}")
    
    # 验证输出是否为正确的类别数
    assert output.prediction_logits.shape[1] == config.num_targets, "输出类别数不正确!"
    
    print("模型测试通过!") 