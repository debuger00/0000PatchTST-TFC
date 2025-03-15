import torch

# Load the .pt file
tensors1 = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\cattle8_lgk\\cattle8_1_c.pt')
tensors2 = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\cattle8_lgk\\cattle8_2_e.pt')






# 合并样本和标签
train_samples = torch.cat((tensors1[0], tensors2[0]), dim=0)

train_samples = train_samples.permute(0, 2, 1);

val_samples = torch.cat((tensors1[1], tensors2[1]), dim=0)

val_samples = val_samples.permute(0, 2, 1);

test_samples = torch.cat((tensors1[2], tensors2[2]), dim=0)

test_samples = test_samples.permute(0, 2, 1);

train_labels = torch.cat((tensors1[3], tensors2[3]), dim=0)
val_labels = torch.cat((tensors1[4], tensors2[4]), dim=0)
test_labels = torch.cat((tensors1[5], tensors2[5]), dim=0)

tensors = [train_samples, val_samples, test_samples, train_labels, val_labels, test_labels]

tensors[0] = tensors[0].squeeze(1);
tensors[0] = tensors[0].permute(0, 2, 1);
tensors[1] = tensors[1].squeeze(1);
tensors[1] = tensors[1].permute(0, 2, 1);
tensors[2] = tensors[2].squeeze(1);
tensors[2] = tensors[2].permute(0, 2, 1);


# Print the type and shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} type: {type(tensor)}')
    if isinstance(tensor, torch.Tensor):
        print(f'Tensor {i+1} shape: {tensor.shape}')
    else:
        print(f'Tensor {i+1} is not a tensor, it is a {type(tensor)}')

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\cattle8_lgk\\01cattle_8.pt');