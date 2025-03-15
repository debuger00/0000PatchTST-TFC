""" train and test dataset

author axiumao
"""

import torch
from torch.utils.data import Dataset
   
class My_Dataset(Dataset):

    def __init__(self, pathway, data_id, transform=None):  
        # pathway是数据集文件路径，data_ id 数据标识符用于区分训练集，测试集和验证集，transform表示对数据进行的预处理
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = torch.load(pathway,weights_only=True)
        if data_id == 0:#训练集
            self.data, self.labels = X_train, Y_train
        elif data_id == 1:
            self.data, self.labels = X_valid, Y_valid
        else:
            self.data, self.labels = X_test, Y_test
        # self.data, self.labels = Data_Segm_random(df_data, n_persegm=40)
        #if transform is given, we transoform data using
        self.transform = transform

    def __len__(self): # 返回集的长度
        return len(self.data)  

    def __getitem__(self, index): # 重载 []
        label = self.labels[index]  #torch.size: [1]         ? 这个labels不是one-hot 编码吗？
        image = self.data[index] #troch.size: [1,50,3]       六个通道， acc_x，acc_y,acc_z,gyr_x,gyr_y,gyr_z
        
        if self.transform:
            image = self.transform(image)

        return image, label 
