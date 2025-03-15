import torch

# Load the .pt file
tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\horse18_lgk\\horse18_lgk_dataset.pt')
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

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\horse18_lgk\\03horse18.pt');
