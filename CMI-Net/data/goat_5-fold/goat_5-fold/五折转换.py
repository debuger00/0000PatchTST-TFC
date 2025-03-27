import torch
tensors = torch.load("E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\goat_5-fold\\goat_5-fold\\new_goat_25hz_3axis_1.pt")

tensors[0] = tensors[0].squeeze(1);
tensors[0] = tensors[0].permute(0, 2, 1);
tensors[1] = tensors[1].squeeze(1);
tensors[1] = tensors[1].permute(0, 2, 1);
tensors[2] = tensors[2].squeeze(1);
tensors[2] = tensors[2].permute(0, 2, 1);

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\001goat.pt')
# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')

tensors = torch.load("E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\goat_5-fold\\goat_5-fold\\new_goat_25hz_3axis_2.pt")

tensors[0] = tensors[0].squeeze(1);
tensors[0] = tensors[0].permute(0, 2, 1);
tensors[1] = tensors[1].squeeze(1);
tensors[1] = tensors[1].permute(0, 2, 1);
tensors[2] = tensors[2].squeeze(1);
tensors[2] = tensors[2].permute(0, 2, 1);

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\002goat.pt')
# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')


tensors = torch.load("E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\goat_5-fold\\goat_5-fold\\new_goat_25hz_3axis_3.pt")

tensors[0] = tensors[0].squeeze(1);
tensors[0] = tensors[0].permute(0, 2, 1);
tensors[1] = tensors[1].squeeze(1);
tensors[1] = tensors[1].permute(0, 2, 1);
tensors[2] = tensors[2].squeeze(1);
tensors[2] = tensors[2].permute(0, 2, 1);

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\003goat.pt')
# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')



tensors = torch.load("E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\goat_5-fold\\goat_5-fold\\new_goat_25hz_3axis_4.pt")

tensors[0] = tensors[0].squeeze(1);
tensors[0] = tensors[0].permute(0, 2, 1);
tensors[1] = tensors[1].squeeze(1);
tensors[1] = tensors[1].permute(0, 2, 1);
tensors[2] = tensors[2].squeeze(1);
tensors[2] = tensors[2].permute(0, 2, 1);

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\004goat.pt')
# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')


tensors = torch.load("E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\goat_5-fold\\goat_5-fold\\new_goat_25hz_3axis_5.pt")

tensors[0] = tensors[0].squeeze(1);
tensors[0] = tensors[0].permute(0, 2, 1);
tensors[1] = tensors[1].squeeze(1);
tensors[1] = tensors[1].permute(0, 2, 1);
tensors[2] = tensors[2].squeeze(1);
tensors[2] = tensors[2].permute(0, 2, 1);

torch.save(tensors, 'E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\005goat.pt')
# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')
