import pandas as pd
import torch
import os
import numpy as np

def load_and_process_data(file_paths):
    """
    加载和处理数据的函数
    
    参数:
        file_paths: 包含多个CSV文件路径的列表
        
    功能:
        1. 遍历所有CSV文件
        2. 读取每个文件中的数据
        3. 删除标签为空的行
        4. 提取三轴加速度数据(AccX, AccY, AccZ)作为特征
        5. 提取对应的标签数据
        
    返回:
        data: 包含所有文件特征数据的列表
        labels: 包含所有文件标签数据的列表
    """
    # 定义标签映射字典
    label_mapping = {
        'RES': 0,  # 休息
        'RUS': 1,  # 起身
        'MOV': 2,  # 移动
        'GRZ': 3,  # 吃草
        'SLT': 4,  # 躺卧
        'FES': 5,  # 采食
        'DRN': 6,  # 饮水
        'LCK': 7,  # 舔舐
        'REL': 8,  # 反刍
        'URI': 9,  # 排尿
        'ATT': 10, # 注意
        'ESC': 11, # 逃避
        'BMN': 12  # 反刍躺卧
    }
    
    data = []  # 存储所有文件的特征数据
    labels = [] # 存储所有文件的标签数据
    
    for file_path in file_paths:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 删除标签为NaN的行
        df = df.dropna(subset=['label'])
        
        # 将字符串标签转换为数值
        df['label'] = df['label'].map(label_mapping)
        
        # 检查是否有未映射的标签
        if df['label'].isna().any():
            unmapped_labels = df[df['label'].isna()]['label'].unique()
            print(f"警告：发现未映射的标签: {unmapped_labels}")
            # 删除未映射的标签行
            df = df.dropna(subset=['label'])
        
        # 提取特征(三轴加速度)和标签
        features = df[['AccX', 'AccY', 'AccZ']].values  # 提取加速度数据作为特征
        label = df['label'].values.astype(int)  # 确保标签为整数类型
        
        # 将当前文件的数据添加到列表中
        data.append(features)
        labels.append(label)
    
    return data, labels

def create_samples(data, labels, window_size=50, overlap=0.5):
    """
    创建滑动窗口样本的函数
    
    参数:
        data: 包含三轴加速度数据的列表
        labels: 对应的标签数据列表  
        window_size: 滑动窗口大小,默认50个时间点
        overlap: 窗口重叠率,默认0.5(50%)
        
    功能:
        1. 使用滑动窗口将连续的传感器数据切分成固定大小的样本
        2. 窗口之间有重叠,通过overlap参数控制重叠程度
        3. 对于每个窗口,选择出现最多次的标签作为该样本的标签
        
    返回:
        samples: 切分后的样本数据列表,每个样本包含window_size个时间点的数据
        sample_labels: 对应的样本标签列表
    """
    samples = []  # 存储切分后的样本
    sample_labels = []  # 存储样本对应的标签
    step = int(window_size * (1 - overlap))  # 计算滑动步长
    
    # 使用滑动窗口切分数据
    for i in range(0, len(data) - window_size + 1, step):
        # 提取当前窗口的数据和标签
        sample = data[i:i + window_size]  # 获取一个窗口的传感器数据
        sample_label = labels[i:i + window_size]  # 获取对应的标签
        
        # 选择窗口内出现最多的标签作为该样本的标签
        unique_labels, counts = np.unique(sample_label, return_counts=True)
        most_frequent_label = unique_labels[np.argmax(counts)]
        
        # 保存样本和对应的标签
        samples.append(sample)
        sample_labels.append(most_frequent_label)
    
    return samples, sample_labels

def main():
    try:
        # Define file paths - 使用os.path.join确保路径正确
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_paths = [
            os.path.join(current_dir, f'cow{i}.csv') for i in range(1, 7)
        ]
        
        # 检查文件是否存在
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"错误：找不到文件 {file_path}")
                return
        
        print("开始加载数据...")
        data, labels = load_and_process_data(file_paths)
        print(f"成功加载 {len(data)} 个文件的数据")
        
        print("开始创建数据集...")
        # Create datasets
        train_data, train_labels = create_samples(
            np.concatenate([data[0], data[1], data[2], data[3]], axis=0),
            np.concatenate([labels[0], labels[1], labels[2], labels[3]], axis=0)
        )
        val_data, val_labels = create_samples(data[4], labels[4])
        test_data, test_labels = create_samples(data[5], labels[5])
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"测试集大小: {len(test_data)}")
        
        # 转换为PyTorch张量
        train_data = torch.FloatTensor(train_data)
        train_labels = torch.LongTensor(train_labels)
        val_data = torch.FloatTensor(val_data)
        val_labels = torch.LongTensor(val_labels)
        test_data = torch.FloatTensor(test_data)
        test_labels = torch.LongTensor(test_labels)
        
        print("数据转换为PyTorch张量完成")
        print(f"训练数据形状: {train_data.shape}")
        print(f"验证数据形状: {val_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        
        # 检查标签的范围
        print(f"标签范围: {torch.min(train_labels).item()} 到 {torch.max(train_labels).item()}")
        print(f"唯一标签值: {torch.unique(train_labels).tolist()}")
        
        # 指定保存路径
        save_path = os.path.join(current_dir, 'cow6_wyh_dataset.pt')
        
        # Save datasets

        torch.save([train_data, train_labels, val_data, val_labels, test_data, test_labels], save_path);
        # torch.save({
        #     'train_data': train_data,
        #     'train_labels': train_labels,
        #     'val_data': val_data,
        #     'val_labels': val_labels,
        #     'test_data': test_data,
        #     'test_labels': test_labels
        # }, save_path)
        
        print(f"数据集已保存到: {save_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
