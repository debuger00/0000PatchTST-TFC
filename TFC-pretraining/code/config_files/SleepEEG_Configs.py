
class Config(object):
    def __init__(self):
        # 模型配置参数
        self.input_channels = 1  # 输入通道数
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

        self.TSlength_aligned = 178  # 对齐后的时间序列长度

        self.CNNoutput_channel = 10  # CNN输出通道数,对于Epilepsy模型为10

        # 训练配置
        self.num_epoch = 40  # 训练轮数

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