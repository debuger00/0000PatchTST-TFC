# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:50:07 2021

@author: axmao2-c
"""

""" train and test dataset

author baiyu
"""

import random
import numpy as np
import torch

import platform
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Data segmentation
def Data_Segm(df_data, single=True, tri=False):
    segments, counts = np.unique(df_data["segment"], return_counts=True)
    samples = []
    labels = []
    window_size = 50  # 25Hz * 2s = 50 samples
    step = 25  # 重叠窗口50%

    for s in segments:
        data_segment = df_data[df_data['segment'] == s]
        sample_persegm = []
        for j in range(0, len(data_segment), step):
            # 提取 Ax, Ay, Az，窗口大小为50
            temp_sample = data_segment[['Ax', 'Ay', 'Az']].iloc[j:j + window_size, :].values
            if len(temp_sample) == window_size:
                sample_persegm.append(temp_sample)
        samples.append(sample_persegm)
        labels.append(list(set(data_segment['label']))[0])

    samples_all = []
    labels_all = []
    for i in range(len(labels)):
        if single:
            for s in samples[i]:
                samples_all.append([s])
                labels_all.append(labels[i])
        if tri:
            for j in range(len(samples[i])):
                if (j + 2) < len(samples[i]):
                    samples_all.append([samples[i][j], samples[i][j + 1], samples[i][j + 2]])
                    labels_all.append(labels[i])

    return samples_all, labels_all


# Get training data, validation data, and test data
def get_data(train_subjects=[2, 3, 7, 8], valid_subject=11, test_subject=14):
    path = 'C:\\Users\\10025\\Desktop\\预训练数据集\\horse_18\\Acc_Gyr_Data.csv'
    df_train_raw = pd.read_csv(path)
    df_train_raw = df_train_raw.drop(['sample_index'], axis=1)

    # 标签映射（保持不变）
    df_train_raw.replace({
        'grazing': 0, 'eating': 0,
        'galloping-natural': 1, 'galloping-rider': 1,
        'standing': 2,
        'trotting-rider': 3, 'trotting-natural': 3,
        'walking-natural': 4, 'walking-rider': 5
    }, inplace=True)

    # 仅选择 Ax, Ay, Az 三列
    feature_columns = ['Ax', 'Ay', 'Az']

    # 数据标准化（仅对 Ax, Ay, Az）
    for col in feature_columns:
        scaler = StandardScaler()
        df_train_raw[col] = scaler.fit_transform(df_train_raw[col].values.reshape(-1, 1)).flatten()

    # 分割数据集
    df_train = df_train_raw.loc[df_train_raw['subject'].isin(train_subjects)]
    df_valid = df_train_raw[df_train_raw['subject'] == valid_subject]
    df_test = df_train_raw[df_train_raw['subject'] == test_subject]

    return df_train, df_valid, df_test


# 生成数据
df_train, df_valid, df_test = get_data(train_subjects=[14, 2, 3, 7], valid_subject=8, test_subject=11)
samples_train, labels_train = Data_Segm(df_train, single=True, tri=False)
samples_valid, labels_valid = Data_Segm(df_valid, single=True, tri=False)
samples_test, labels_test = Data_Segm(df_test, single=True, tri=False)

# 转换为Tensor并保存
tensor_samples_train = torch.from_numpy(np.array(samples_train)).float()
tensor_label_train = torch.from_numpy(np.array(labels_train)).type(torch.LongTensor)

tensor_samples_valid = torch.from_numpy(np.array(samples_valid)).float()
tensor_label_valid = torch.from_numpy(np.array(labels_valid)).type(torch.LongTensor)

tensor_samples_test = torch.from_numpy(np.array(samples_test)).float()
tensor_label_test = torch.from_numpy(np.array(labels_test)).type(torch.LongTensor)

torch.save(
    [tensor_samples_train, tensor_samples_valid, tensor_samples_test, tensor_label_train, tensor_label_valid,
     tensor_label_test],
    "./horse18.pt"  # 修改保存文件名以区分
)
print(tensor_samples_train.shape)  # 应为 (样本数, 1, 50, 3)

plt.plot(tensor_samples_train[0][0][:, 0])  # 第一个样本的Ax数据
plt.title("Ax数据示例（25Hz, 2s窗口）")
plt.show()
