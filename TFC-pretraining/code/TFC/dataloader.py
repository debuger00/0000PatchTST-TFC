import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft

def generate_freq(dataset, config):
    X_train = dataset["samples"]
    y_train = dataset['labels']
    # shuffle
    data = list(zip(X_train, y_train))
    np.random.shuffle(data)
    data = data[:10000] # take a subset for testing.
    X_train, y_train = zip(*data)
    X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

    """Align the TS length between source and target datasets"""
    X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

    if isinstance(X_train, np.ndarray):
        x_data = torch.from_numpy(X_train)
    else:
        x_data = X_train

    """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
    the output shape is half of the time window."""

    x_data_f = fft.fft(x_data).abs() #/(window_length) # rfft for real value inputs. 将时域数据转换为频域数据 .abs() 取绝对值,
    return (X_train, y_train, x_data_f)

class Load_Dataset(Dataset):
    """
    数据加载器类,继承自PyTorch的Dataset类。
    主要功能:
    1. 加载和预处理时序数据
    2. 对数据进行对齐和格式转换
    3. 在预训练模式下进行数据增强
    
    参数:
        dataset: 包含samples和labels的字典
        config: 配置参数
        training_mode: 训练模式('pre_train'或其他)
        target_dataset_size: 目标数据集大小,默认64
        subset: 是否使用子集进行调试,默认False
    """
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__() # # 传统写法等价于 super().__init__()
        self.training_mode = training_mode
        
        # 加载数据并打乱
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        # 确保数据是3维的[batch, channel, time]格式
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)

        # 对齐不同数据集的时序长度
        X_train = X_train[:, :3, :int(config.TSlength_aligned)]

        # 如果使用子集进行调试
        if subset == True:
            subset_size = target_dataset_size * 10
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            print('Using subset for debugging, the datasize is:', y_train.shape[0])

        # 转换数据类型
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        # 将时域数据转换到频域
        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs()
        self.len = X_train.shape[0]

        # 在预训练模式下进行数据增强
        if training_mode == "pre_train":
            self.aug1 = DataTransform_TD(self.x_data, config)  # 时域增强
            self.aug1_f = DataTransform_FD(self.x_data_f, config)  # 频域增强

    def __getitem__(self, index):
        # 根据训练模式返回不同的数据形式
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset=True):
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    # finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))  # train.pt
    # test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))  # test.pt
    """In pre-training: 
    train_dataset: [371055, 1, 178] from SleepEEG.    
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset) # for self-supervised, the data are augmented here
    # finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)
    # test_dataset = Load_Dataset(test_dataset, configs, training_mode,
    #                             target_dataset_size=configs.target_batch_size, subset=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    # finetune_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
    #                                            shuffle=True, drop_last=configs.drop_last,
    #                                            num_workers=0)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
    #                                           shuffle=True, drop_last=False,
    #                                           num_workers=0)

    # return train_loader, finetune_loader, test_loader
    return train_loader