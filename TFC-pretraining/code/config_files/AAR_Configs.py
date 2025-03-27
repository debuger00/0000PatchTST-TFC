
class Config(object):
    def __init__(self):
        # 模型配置参数
        self.input_channels = 3 # 输入通道数
        self.increased_dim = 1   # 增加的维度
        self.final_out_channels = 128  # 最终输出通道数
        self.num_classes = 5     # 源域分类数量
        self.num_classes_target = 2  # 目标域分类数量 
        self.dropout = 0.35      # dropout比率

        # CNN参数
        self.kernel_size = 25    # 卷积核大小
        self.stride = 3          # 步长
        self.features_len = 127  # 特征长度
        self.features_len_f = self.features_len  # 特征长度(用于fine-tune)

        self.TSlength_aligned = 50  # 对齐后的时间序列长度

        self.CNNoutput_channel = 10  # CNN输出通道数,对于Epilepsy模型为10

        # 训练配置
        self.num_epoch = 200  # 训练轮数

        # 优化器参数
        self.optimizer = 'adam'  # 优化器类型
        self.beta1 = 0.9        # Adam优化器的beta1参数
        self.beta2 = 0.99       # Adam优化器的beta2参数
        self.lr = 3e-4          # 学习率
        self.lr_f = self.lr     # fine-tune时的学习率

        # 数据参数
        self.drop_last = True    # 是否丢弃最后不完整的batch
        self.batch_size = 128    # 训练batch大小
        """对于Epilepsy数据集,目标batch大小为60"""
        self.target_batch_size = 60   # 目标数据集batch大小(用于fine-tune的样本数)

        # 实例化其他配置类
        self.Context_Cont = Context_Cont_configs()  # 上下文对比学习配置
        self.TC = TC()                             # 时间对比学习配置
        self.augmentation = augmentations()        # 数据增强配置

############################################################################################################
        # # 基础参数
        # self.enc_in = 3  # 输入特征维度
        # self.seq_len = 50  # 输入序列长度
        # self.pred_len = 0  # 分类任务不需要预测长度
        
        # # 模型结构参数
        # self.e_layers = 1  # encoder层数
        # self.n_heads = 8  # 注意力头数
        # self.d_model = 128  # 模型维度
        # self.d_ff = 256  # 前馈网络维度
        # self.dropout = 0.1  # dropout率
        # self.fc_dropout = 0.5  # 全连接层dropout率
        # self.head_dropout = 0.1  # 输出头dropout率
        
        # # Patch相关参数
        # self.patch_len = 6  # patch长度
        # self.stride = 3  # patch步长
        # self.padding_patch = 'end'  # patch填充方式
        
        # # 其他参数
        # self.individual = False  # 是否独立处理每个特征
        # self.num_classes = 5  # 分类类别数


   
   
    #      # 基础参数
    #     # 基础参数
    #     self.enc_in = 3         # 输入特征维度
    #     self.seq_len = 50        # 输入序列长度
    #     self.pred_len = 24       # 预测序列长度
        
    #     # 模型结构参数
    #     self.e_layers = 2        # encoder层数
    #     self.n_heads = 8         # 注意力头数
    #     self.d_model = 128       # 模型维度
    #     self.d_ff = 256         # 前馈网络维度
    #     self.dropout = 0.2       # dropout率
    #     self.fc_dropout = 0.2    # 全连接层dropout率
    #     self.head_dropout = 0.2  # 输出头dropout率
        
    #     # Patch相关参数 
    #     self.patch_len = 8      # patch长度
    #     self.stride = 4          # patch步长
    #     self.padding_patch = 'end'  # patch填充方式
        
    #     # 1D卷积参数
    #     self.conv1d_kernel_size = 3  # 1D卷积核大小
    #     self.conv1d_out_channels = 32  # 1D卷积输出通道数
        
    #     # 数据处理参数
    #     self.individual = False   # 是否独立处理每个特征
    #     self.revin = False        # 是否使用RevIN
    #     self.affine = False       # RevIN是否使用affine变换
    #     self.subtract_last = False  # 是否减去最后一个值
        
    #     # 分解相关参数
    #     self.decomposition = False  # 是否使用分解
    #     self.kernel_size = 25      # 分解核大小

    #  # 修改预测相关参数为分类参数
    #     self.num_classes = 5     # 分类类别数
        
        
    #     # 分类器特定参数
    #     self.classifier_dropout = 0.5  # 分类器dropout率
    #     self.use_weighted_loss = False # 是否使用加权损失（处理类别不平衡）

    #     # 1D卷积参数
    #     self.conv1d_kernel_size = 3  # 1D卷积核大小
    #     self.conv1d_out_channels = 32  # 1D卷积输出通道数

        
    #     # 计算特征维度 - 确保结果是整数
    #     # 公式1: patch个数 = 取整（（输入序列长度-patch长度）/步长）+ 2
    #     patch_num = int((self.seq_len - self.patch_len) / self.stride) + 2
    #     # 公式2: 特征维度 = patch个数 * 模型维度
    #     self.feature_dim = int(patch_num * self.d_model)  # 特征维


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5  # 抖动缩放比例
        self.jitter_ratio = 2          # 抖动比例
        self.max_seg = 12             # 最大分段数

class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2              # 温度参数
        self.use_cosine_similarity = True   # 是否使用余弦相似度

class TC(object):
    def __init__(self):
        self.hidden_dim = 64    # 隐藏层维度
        self.timesteps = 50     # 时间步长