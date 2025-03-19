from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from PatchTST import PatchTSTNet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config_files.SleepEEG_Configs_PatchTST import Config
from config_files.AAR_Configs import Config

# """Two contrastive encoders"""
# class TFC(nn.Module):
#     def __init__(self, configs):
#         super(TFC, self).__init__()

#         encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, )
#         self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

#         self.projector_t = nn.Sequential(
#             nn.Linear(configs.TSlength_aligned, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#         encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2,)
#         self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

#         self.projector_f = nn.Sequential(
#             nn.Linear(configs.TSlength_aligned, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )


#     def forward(self, x_in_t, x_in_f):
#         """Use Transformer"""
#         x = self.transformer_encoder_t(x_in_t)  # 输入形状是 [batch_size, channel, seq_len]
#         h_time = x.reshape(x.shape[0], -1)  # 将transformer编码器的输出张量重塑为二维张量 [batch_size, features=seq_len *channel]

#         """Cross-space projector"""
#         z_time = self.projector_t(h_time)

#         """Frequency-based contrastive encoder"""
#         f = self.transformer_encoder_f(x_in_f)
#         h_freq = f.reshape(f.shape[0], -1)

#         """Cross-space projector"""
#         z_freq = self.projector_f(h_freq)

#         return h_time, z_time, h_freq, z_freq


"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()
        
    
        self.encoder_t = PatchTSTNet(configs)
        self.encoder_f = PatchTSTNet(configs)
        
        # 保持原有的投影头
        self.projector_t = nn.Sequential(
            nn.Linear(1536, 512 ),  # 使用d_model作为输入维度
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.projector_f = nn.Sequential(
            nn.Linear(1536, 512),  # 使用d_model作为输入维度
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x_in_t, x_in_f):
  
        
        """Time-based encoder"""
        # 输入形状是 [batch_size, channel, seq_len]
        # print(f"000时间域特征 h_time shape: {x_in_t.shape}")  # 应该是 [128, 3,50]
        h_time = self.encoder_t(x_in_t)  
        #输出 h_time shape: torch.Size([128, 1, 178])
        # print(f"001时间域特征 h_time shape: {h_time.shape}")

        h_time = h_time.flatten(1)  # [batch_size, feature_dim]  等价于 h_time = h_time.reshape(h_time.shape[0], -1)
        # print(f"001时间域特征 h_time shape: {h_time.shape}")  # 应该是 [128, 2816]
        """Cross-space projector"""
        z_time = self.projector_t(h_time)
        
        """Frequency-based encoder"""
        h_freq = self.encoder_f(x_in_f)  # 不需要permute了
        h_freq = h_freq.flatten(1)  # [batch_size, feature_dim]
        
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)
        
        return h_time, z_time, h_freq, z_freq
    

"""下游分类器  测试用"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256是因为z_time和z_freq各128维拼接
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, configs.num_classes_target)
        )
    
    def forward(self, x):
        return self.classifier(x)

def test_model_shapes():
    # 使用 SleepEEG_Configs_PatchTST 的配置
    configs = Config()
    model = TFC(configs)
    classifier = target_classifier(configs)
    
    # 创建模拟输入数据
    batch_size = configs.batch_size  # 128
    channel = configs.channel  # 1
    # 输入形状应该是 [batch_size, channel, seq_len]
    x_time = torch.randn(batch_size, channel, configs.TSlength_aligned)  # [128, 1, 178]
    x_freq = torch.randn(batch_size, channel, configs.TSlength_aligned)  # [128, 1, 178]
    
    print("=== 输入数据形状 ===")
    print(f"批次大小: {batch_size}")
    print(f"时间序列长度: {configs.TSlength_aligned}")
    print(f"时间域输入 x_time shape: {x_time.shape}")
    print(f"频率域输入 x_freq shape: {x_freq.shape}")
    
    # 获取模型输出并打印形状
    with torch.no_grad():
        h_time, z_time, h_freq, z_freq = model(x_time, x_freq)
        
        print("\n=== TFC模型各层输出形状 ===")
        print(f"时间域特征 h_time shape: {h_time.shape}")  # 应该是 [128, 2816]
        print(f"时间域投影 z_time shape: {z_time.shape}")  # 应该是 [128, 128]
        print(f"频率域特征 h_freq shape: {h_freq.shape}")  # 应该是 [128, 2816]
        print(f"频率域投影 z_freq shape: {z_freq.shape}")  # 应该是 [128, 128]
        
        # 测试下游分类器
        combined_features = torch.cat([z_time, z_freq], dim=1)  # [128, 256]
        predictions = classifier(combined_features)  # [128, num_classes]
        
        print("\n=== 下游分类器输出形状 ===")
        print(f"组合特征 combined_features shape: {combined_features.shape}")
        print(f"最终分类输出 predictions shape: {predictions.shape}")

if __name__ == "__main__":
    test_model_shapes()