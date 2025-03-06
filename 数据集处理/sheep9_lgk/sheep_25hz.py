import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def interpolate_sample(sample):
    """三次样条插值将25点样本转换为50点"""
    original_time = np.linspace(0, 1.92, num=25)  # 原12.5Hz时间轴
    new_time = np.linspace(0, 1.96, num=50)  # 新25Hz时间轴
    interpolated = np.zeros((50, 3))

    for i in range(3):  # 对x,y,z分别插值
        cs = CubicSpline(original_time, sample[:, i])
        interpolated[:, i] = cs(new_time)
    return interpolated


def Data_Segm(df_data):
    """改进后的数据分段函数（含插值）"""
    segments = np.unique(df_data["seg"])
    samples = []
    labels = []

    for s in segments:
        data_segment = df_data[df_data['seg'] == s]
        sample_persegm = []

        # 滑动窗口处理（步长保持12）
        for j in range(0, len(data_segment), 12):
            temp_sample = data_segment[['x', 'y', 'z']].iloc[j:j + 25].values
            if len(temp_sample) == 25:
                # 执行插值处理
                interpolated = interpolate_sample(temp_sample)
                sample_persegm.append(interpolated)

        samples.extend(sample_persegm)
        labels.extend([data_segment['activity'].iloc[0]] * len(sample_persegm))

    return np.array(samples), np.array(labels)


def process_file(path):
    """处理单个文件的完整流程"""
    # 读取数据
    df = pd.read_csv(path).drop(['sample_index'], axis=1)

    # 数据标准化（按文件独立处理）
    for col in ['x', 'y', 'z']:
        df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1, 1))

    # 数据分段和插值
    samples, labels = Data_Segm(df)

    # 分层分割数据集
    # 第一次分割：训练+临时集（80%训练，20%临时）
    X_train, X_temp, y_train, y_temp = train_test_split(
        samples, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 第二次分割：验证+测试（各取临时集的50%）
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    # 转换为PyTorch张量
    return [
        torch.from_numpy(X_train).float().squeeze(1),  # 去除冗余维度 (N,1,50,3) -> (N,50,3)
        torch.from_numpy(X_val).float().squeeze(1),
        torch.from_numpy(X_test).float().squeeze(1),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(y_val).long(),
        torch.from_numpy(y_test).long()
    ]


# 处理两个文件并保存
paths = [
    'C:\\Users\\10025\\Desktop\\Sheep\\updated_data\\df_train_raw_1.csv',
    'C:\\Users\\10025\\Desktop\\Sheep\\updated_data\\df_train_raw_1.csv'
]

for idx, path in enumerate(paths, start=1):
    # 处理数据
    data = process_file(path)

    # 验证数据维度
    print(f"File {idx} shapes:")
    print(f"Train: {data[0].shape}")
    print(f"Valid: {data[1].shape}")
    print(f"Test : {data[2].shape}")

    # 保存为PT文件
    torch.save(data, f"sheep_data_{idx}.pt")