from utils import get_network, get_mydataloader, get_weighted_mydataloader
from conf import settings
from Class_balanced_loss import CB_loss
from Regularization import Regularization
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F

import time
import argparse
import platform
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import yaml
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, precision_score, recall_score, \
    confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc


class CBLossConfig:
    def __init__(self, num_classes, loss_type, beta, gamma):
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor  # 'val_loss' 或 'val_acc'

    def __call__(self, score):
        # 对于损失，越小越好；对于准确率，越大越好
        if self.monitor == 'val_loss':
            score_improved = self.best_score is None or score < self.best_score - self.min_delta
        else:  # 'val_acc'
            score_improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if self.best_score is None:
            self.best_score = score
        elif not score_improved:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls, device):
    start = time.time()
    network.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        try:
            outputs = network(images)
            loss_type = "focal"

            # 计算类别平衡损失(Class Balanced Loss)
            loss_cb = CB_loss(labels, outputs, samples_per_cls, 5, loss_type, args.beta, args.gamma)

                # 计算交叉熵损失
            loss_ce = loss_function(outputs, labels)
            # 组合损失(这里CB loss的权重为0，实际上只使用了CE loss)
            # loss = 1.0 * loss_ce + 0.0 * loss_cb
            loss = 0.25*loss_ce + 0.75*loss_cb # class-balanced focal loss (CMI-Net+CB focal loss)
 
            if args.weight_d > 0:
                loss += reg_loss(network)

            loss.backward()


        except RuntimeError as e:
            print(f"Warning: {str(e)}")
            optimizer.zero_grad()
            continue

        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += len(labels)
        total_loss += loss.item()

    # 计算整个epoch的平均准确率和损失
    epoch_accuracy = total_correct / total_samples  # 保持为小数形式
    epoch_loss = total_loss / len(train_loader)

    print('Training Epoch: {epoch} [{total}/{total}]\tTrain_accuracy: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        epoch_accuracy,
        epoch_loss,
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        total=len(train_loader.dataset)
    ))

    return network, epoch_accuracy, epoch_loss


@torch.no_grad()
def eval_training(valid_loader, net, samples_per_cls, cb_config, device, epoch=0):
    start = time.time()
    net.eval()

    n = 0
    valid_loss = 0.0
    correct = 0.0
    class_target = []
    class_predict = []

    # 在 GPU 上收集所有数据，减少频繁的 GPU 到 CPU 转换
    for (images, labels) in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        # 使用与训练相同的损失函数
        if args.use_mixed_loss == 1:
            loss = mixed_loss(
                labels, outputs, samples_per_cls, args.num_classes, 
                'focal', args.beta, args.gamma, args.dice_weight
            )
        else:
            loss = CB_loss(
                labels, outputs, samples_per_cls, args.num_classes, 
                cb_config.loss_type, args.beta, args.gamma
            )

        valid_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        # 收集 GPU 上的张量
        class_target.append(labels)
        class_predict.append(preds)

        n += 1

    # 在所有数据收集完之后再进行转换到 CPU
    class_target = torch.cat(class_target).cpu().numpy().tolist()
    class_predict = torch.cat(class_predict).cpu().numpy().tolist()

    # 打印分类报告
    report = classification_report(class_target, class_predict, zero_division=0)
    print('Evaluating Network.....')
    print('Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
        epoch,
        valid_loss / n,
        correct.float() / len(valid_loader.dataset),
        time.time() - start
    ))
    print('Classification Report:')
    print(report)

    # 打印混淆矩阵
    print('\nConfusion Matrix - Epoch {}:'.format(epoch))
    matrix = confusion_matrix(class_target, class_predict)

    # 打印列索引
    n_classes = len(matrix)
    print('\t' + '\t'.join([str(i) for i in range(n_classes)]))
    print('-' * (8 * n_classes))

    # 打印每一行
    for i in range(n_classes):
        row = [str(x) for x in matrix[i]]
        print(f'{i}\t' + '\t'.join(row))
    print()  # 添加空行以提高可读性

    accuracy = correct.float() / len(valid_loader.dataset)
    avg_loss = valid_loss / len(valid_loader.dataset)
    f1 = f1_score(class_target, class_predict, average='macro', zero_division=0)

    # 确保返回的是CPU张量
    return accuracy.cpu(), avg_loss, f1, class_predict


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def save_model_config(model, args, train_config, save_path):
    """保存模型配置到YAML文件"""
    config = {
        'model_name': args.net,
        'model_architecture': {
            'layers': [],
            'total_parameters': 0,
            'trainable_parameters': 0
        },
        'training_config': {
            'batch_size': args.b,
            'learning_rate': args.lr,
            'epochs': args.epoch,
            'seed': args.seed,
            'weight_decay': args.weight_d,
            'beta': args.beta,
            'gamma': args.gamma,
            'gpu': bool(args.gpu),
            'optimizer': 'AdamW',
            'loss_function': 'Mixed Loss (CB_loss + Dice)' if args.use_mixed_loss == 1 else 'CB_loss with focal',
            'dice_weight': args.dice_weight if args.use_mixed_loss == 1 else 0,
            'scheduler': 'LambdaLR with warmup and cosine decay'
        },
        'performance_metrics': {
            'best_accuracy': 0,
            'best_epoch': 0,
            'final_metrics': {
                'accuracy': 0,
                'f1_score': 0,
                'precision': 0,
                'recall': 0,
                'kappa': 0
            }
        }
    }

    # 记录模型层结构
    for name, module in model.named_children():
        layer_info = {
            'name': name,
            'type': module.__class__.__name__,
            'parameters': sum(p.numel() for p in module.parameters())
        }

        # 对于特定类型的层，添加更多详细信息
        if isinstance(module, nn.Linear):
            layer_info.update({
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None
            })
        elif isinstance(module, nn.Conv2d):
            layer_info.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })

        config['model_architecture']['layers'].append(layer_info)

    # 记录参数总量
    total_params, trainable_params = get_parameter_number(model)
    config['model_architecture']['total_parameters'] = total_params
    config['model_architecture']['trainable_parameters'] = trainable_params

    # 保存配置到YAML文件
    config_path = os.path.join(save_path, 'model_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# 添加混合损失函数
def dice_loss(labels_one_hot, logits, smooth=1.0, weights=None):
    """
    计算Dice Loss，特别适合处理类别不平衡问题。
    
    Dice系数衡量两个集合的相似度，Dice Loss = 1 - Dice系数，公式为:
    DL = 1 - (2*|X∩Y| + smooth) / (|X|+|Y| + smooth)
    其中X是预测结果，Y是真实标签。
    
    参数:
        labels_one_hot (torch.Tensor): 形状为[batch, no_of_classes]的one-hot编码标签。
        logits (torch.Tensor): 形状为[batch, no_of_classes]的模型原始输出分数。
        smooth (float): 平滑系数，防止分母为0。
        weights (torch.Tensor, optional): 各类别权重，形状为[no_of_classes]。
        
    返回:
        torch.Tensor: 标量张量，表示计算得到的Dice Loss值。
    """
    # 将logits转换为概率
    probs = torch.softmax(logits, dim=1)
    
    # 计算每个类别的Dice系数
    numerator = 2.0 * torch.sum(probs * labels_one_hot, dim=0) + smooth
    denominator = torch.sum(probs, dim=0) + torch.sum(labels_one_hot, dim=0) + smooth
    dice_coef = numerator / denominator
    
    # 若指定权重，则应用权重
    if weights is not None:
        dice_coef = dice_coef * weights
        dice_loss = 1.0 - torch.sum(dice_coef) / torch.sum(weights)
    else:
        dice_loss = 1.0 - torch.mean(dice_coef)
    
    return dice_loss

def mixed_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, dice_weight=0.3):
    """
    结合CB Loss和Dice Loss的混合损失函数，增强对少数类的学习。
    
    参数:
        labels (torch.Tensor): 形状为[batch]的整数张量，包含每个样本的类别标签。
        logits (torch.Tensor): 形状为[batch, no_of_classes]的浮点张量，包含模型的原始输出分数。
        samples_per_cls (list/torch.Tensor): 大小为[no_of_classes]的列表，包含每个类别的样本总数。
        no_of_classes (int): 类别总数。
        loss_type (str): CB Loss使用的基础损失函数类型，可选"sigmoid"、"focal"或"softmax"。
        beta (float): 类别平衡的超参数，用于计算有效样本数，通常接近1。
        gamma (float): Focal Loss的聚焦参数，仅在loss_type="focal"时使用。
        dice_weight (float): Dice Loss在混合损失中的权重，范围[0,1]，默认0.3。
        
    返回:
        torch.Tensor: 标量张量，表示计算得到的混合损失值。
    """
    device = logits.device
    
    # 计算CB Loss
    cb_loss_val = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)
    
    # 转换为one-hot编码
    labels_one_hot = F.one_hot(labels, no_of_classes).float().to(device)
    
    # 计算类别权重，确保样本数越少的类别权重越大
    if not isinstance(samples_per_cls, torch.Tensor):
        samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float)
    samples_per_cls = samples_per_cls.to(device)
    
    # 使用倒数作为权重，样本越少权重越大
    dice_cls_weights = 1.0 / (samples_per_cls + 1.0)  # 添加1防止除零
    dice_cls_weights = dice_cls_weights / dice_cls_weights.sum() * no_of_classes
    
    # 计算Dice Loss
    dice_loss_val = dice_loss(labels_one_hot, logits, weights=dice_cls_weights)
    
    # 组合两种损失
    total_loss = (1.0 - dice_weight) * cb_loss_val + dice_weight * dice_loss_val
    
    return total_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST', help='net type')
    parser.add_argument('--gpu', type=int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoches')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--gamma', type=float, default=1.0, help='the gamma of focal loss')
    parser.add_argument('--beta', type=float, default=0.99, help='the beta of class balanced loss')
    parser.add_argument('--weight_d', type=float, default=0.01, help='weight decay for regularization')
    parser.add_argument('--save_path', type=str, default='experiments/default_run',
                        help='path for saving all outputs (checkpoints, logs, etc)')
    # parser.add_argument('--data_path', type=str,
    #                     default='C:\\Users\\10025\\Desktop\\0000PatchTST-TFC-main\\0000PatchTST-TFC-main\\CMI-Net\\data\\new_goat_25hz_3axis.pt',
    #                     help='saved path of input data')
    parser.add_argument('--data_path', type=str,
                        default='/data1/wangyonghua/0000PatchTST-TFC/CMI-Net/data/new_goat_25hz_3axis.pt',
                        help='saved path of input data')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes in the dataset')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm for gradient clipping')
    parser.add_argument('--num_workers', type=int, default=8 if platform.system() != "Windows" else 0,
                        help='number of workers for data loading')
    parser.add_argument('--lr_decay', type=str, default='cosine',
                        help='learning rate decay type: cosine/step/linear/cyclic (default: cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--cycle_epochs', type=int, default=10, help='number of epochs per cycle for cyclic lr')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='device to use')
    parser.add_argument('--use_mixed_loss', type=int, default=0, help='whether to use mixed loss (CB Loss + Dice Loss): 0 for CB Loss only, 1 for mixed loss')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='weight of Dice Loss in mixed loss')
    parser.add_argument('--weight_smooth', type=int, default=1, help='whether to apply weight smoothing: 0 for no smoothing, 1 for log smoothing')
    parser.add_argument('--class_balanced_sampling', type=int, default=0, help='whether to use class balanced sampling: 0 for no, 1 for yes')
    parser.add_argument('--focal_alpha', type=float, default=None, help='alpha parameter for focal loss, if None, use class weights')

    args = parser.parse_args()

    # 在创建网络前设置device参数
    args.device = torch.device(args.device if torch.cuda.is_available() and args.gpu else "cpu")

    def get_lr_scheduler(optimizer, args):
        if args.lr_decay == 'cosine':
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return epoch / args.warmup_epochs
                else:
                    # 确保学习率不低于最小学习率
                    progress = (epoch - args.warmup_epochs) / (args.epoch - args.warmup_epochs)
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    return max(cosine_decay, args.min_lr / args.lr)
        elif args.lr_decay == 'step':
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return epoch / args.warmup_epochs
                else:
                    # 每30个epoch衰减为原来的0.1
                    return 0.1 ** (epoch // 30)
        elif args.lr_decay == 'cyclic':
            def lr_lambda(epoch):
                # 计算当前周期内的位置
                cycle_position = epoch % args.cycle_epochs
                # 计算周期内的相对位置 (0 到 1)
                relative_position = cycle_position / args.cycle_epochs
                
                # 在周期的前半部分增加学习率，后半部分减少学习率
                if relative_position < 0.5:
                    # 前半部分：从1.0到2.0线性增加
                    return 1.0 + 2.0 * relative_position
                else:
                    # 后半部分：从2.0到1.0线性减少
                    return 2.0 - 2.0 * (relative_position - 0.5)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"使用周期性学习率调整，周期长度: {args.cycle_epochs} epochs")
        return scheduler

    # 修改保存路径逻辑
    # 创建保存目录
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # 创建时间戳子目录 (格式: 日d_时h_分m)
    timestamp = time.strftime("%dd_%Hh_%Mm")
    checkpoint_path = os.path.join(args.save_path, timestamp)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # 创建子目录
    checkpoints_dir = os.path.join(checkpoint_path, 'checkpoints')
    logs_dir = os.path.join(checkpoint_path, 'logs')
    plots_dir = os.path.join(checkpoint_path, 'plots')

    # 创建所需的子目录
    for dir_path in [checkpoints_dir, logs_dir, plots_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 修改checkpoint_path_pth的路径
    checkpoint_path_pth = os.path.join(checkpoints_dir, '{net}-{type}.pth')

    # 创建配置对象
    cb_config = CBLossConfig(
        num_classes=args.num_classes,
        loss_type="focal",
        beta=args.beta,
        gamma=args.gamma
    )

    # 在训练循环开始前定义设备
    device = args.device
    net = get_network(args)

    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpu = 0
        torch.manual_seed(args.seed)

    # 修改工作进程数设置部分
    sysstr = platform.system()
    num_workers = args.num_workers

    pathway = args.data_path
    if sysstr == 'Linux':
        pathway = args.data_path

    # 创建数据加载器
    train_loader, weight_train, number_train = get_weighted_mydataloader(
        pathway,
        data_id=0,
        batch_size=args.b,
        num_workers=num_workers,
        shuffle=True
    )
    valid_loader = get_mydataloader(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)

    # 打印模型结构详情
    print("\n" + "=" * 50)
    print("模型结构详情：")
    print(net)

    # 特别检查输出层
    print("\n输出层信息:")


    def get_output_dim(model):
        """获取模型输出维度的辅助函数"""
        # 首先检查分类器
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                # 获取Sequential中的最后一个Linear层
                for layer in reversed(model.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
            elif isinstance(model.classifier, nn.Linear):
                return model.classifier.out_features

        # 如果上述方法都失败，尝试遍历整个模型找到最后的Linear层
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is not None:
            return last_linear.out_features

        raise AttributeError("无法找到模型的输出维度！")


    try:
        # 获取数据集的类别数
        if hasattr(train_loader.dataset, 'classes'):
            num_classes = len(train_loader.dataset.classes)
        else:
            # 如果没有classes属性，尝试从数据中推断类别数
            all_labels = []
            for _, labels in train_loader:
                all_labels.extend(labels.numpy())
            num_classes = len(np.unique(all_labels))
        print(f"数据集的类别数: {num_classes}")

        # 获取模型输出维度
        try:
            output_dim = get_output_dim(net)
            print(f"模型输出层维度: {output_dim}")

            if output_dim == num_classes:
                print("✓ 模型输出维度与数据类别数匹配")
            else:
                print(f"警告：模型输出层维度({output_dim})与数据类别数({num_classes})不匹配！")
        except AttributeError as e:
            print(f"警告：获取模型输出维度时出错 - {str(e)}")

    except Exception as e:
        print(f"警告：检查类别数时出现问题: {str(e)}")

    print("=" * 50 + "\n")

    # 将 weight_train 移动到正确的设备上
    if isinstance(weight_train, torch.Tensor):
        weight_train = weight_train.to(device)

    # 确保 number_train 也在正确的设备上
    if isinstance(number_train, torch.Tensor):
        number_train = number_train.to(device)

    if args.weight_d > 0:
        reg_loss = Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")

    # 修改优化器参数
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_d)
    train_scheduler = get_lr_scheduler(optimizer, args)

    # 修改早停策略 - 基于验证准确率
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
        verbose=True,
        monitor='val_acc'  # 明确指定监控验证准确率
    )

    # 获取类别计数
    def get_class_counts(train_loader):
        class_counts = torch.zeros(args.num_classes)
        for _, labels in train_loader:
            for i in range(args.num_classes):
                class_counts[i] += (labels == i).sum().item()
        return class_counts

    # 计算类别权重
    class_counts = get_class_counts(train_loader)
    samples_per_cls = class_counts
    
    # 计算有效样本数和初始权重
    effective_num = 1.0 - torch.pow(args.beta, samples_per_cls)
    weights = (1.0 - args.beta) / effective_num
    
    # 计算权重比例并打印
    max_weight = weights.max()
    min_weight = weights.min()
    weight_ratio = max_weight / min_weight
    print(f"\n原始权重比例: {weight_ratio:.2f}")
    
    # 如果权重比例过大，应用权重平滑
    if args.weight_smooth == 1 and weight_ratio > 10.0:
        print("权重比例过大，应用权重平滑...")
        # 应用对数平滑
        weights = torch.log1p(weights)
        # 重新归一化
        weights = weights / weights.sum() * len(weights)
        # 重新计算权重比例
        max_weight = weights.max()
        min_weight = weights.min()
        weight_ratio = max_weight / min_weight
        print(f"平滑后的权重比例: {weight_ratio:.2f}")
    
    # 打印类别权重
    print(f"\nClass weights with beta = {args.beta}")
    for i in range(args.num_classes):
        print(f"Class {i}: {weights[i]:.4f}")
    
    # 如果权重差异仍然很大，发出警告
    if weight_ratio > 10.0:
        print("\n警告: 类别权重差异过大，可能导致训练不稳定。考虑降低beta值。")
    
    # 打印gamma值
    print(f"\nFocal Loss gamma = {args.gamma}")
    if args.gamma > 2.0:
        print("警告: gamma值较大，可能导致对难样本过度关注，训练不稳定。")

    # 使用CB_loss作为统一的损失函数
    cb_config = CBLossConfig(args.num_classes, 'focal', args.beta, args.gamma)
    
    # 根据参数选择使用混合损失还是CB Loss
    if args.use_mixed_loss == 1:
        print(f"\n使用混合损失函数 (CB Loss + Dice Loss)")
        print(f"Dice Loss权重: {args.dice_weight}")
        # 创建混合损失函数
        criterion = lambda outputs, labels: mixed_loss(
            labels, outputs, samples_per_cls, args.num_classes, 
            'focal', args.beta, args.gamma, args.dice_weight
        )
    else:
        print(f"\n使用CB Loss ({cb_config.loss_type})")
        print(f"Beta: {args.beta}, Gamma: {args.gamma}")
        # 创建CB Loss函数
        criterion = lambda outputs, labels: CB_loss(
            labels, outputs, samples_per_cls, args.num_classes, 
            cb_config.loss_type, args.beta, args.gamma
        )

    # 添加类别平衡采样器
    if args.class_balanced_sampling == 1:
        print("\n使用类别平衡采样...")
        # 获取训练集中每个样本的类别
        train_labels = []
        # 创建临时数据加载器来获取所有标签
        temp_loader = torch.utils.data.DataLoader(
            train_loader.dataset, 
            batch_size=1000, 
            shuffle=False, 
            num_workers=args.num_workers
        )
        for _, labels in temp_loader:
            train_labels.append(labels)
        train_labels = torch.cat(train_labels)
        
        # 计算每个类别的权重
        class_sample_count = torch.bincount(train_labels)
        weight = 1. / class_sample_count.float()
        
        # 为每个样本分配权重
        samples_weight = weight[train_labels]
        
        # 创建WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(train_labels),
            replacement=True
        )
        
        # 使用采样器创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=args.b,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print("已启用类别平衡采样")
    else:
        # 使用普通数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=args.b,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # 初始化记录列表
    Train_Accuracy = []
    Train_Loss = []
    Valid_Accuracy = []
    Valid_Loss = []
    f1_s = []
    best_val_acc = 0
    best_acc = 0
    best_epoch = 0

    # 定义最佳模型保存路径
    best_weights_path = os.path.join(checkpoints_dir, f'{args.net}-best.pth')

    # 开始正式训练循环...
    for epoch in range(1, args.epoch + 1):
        # 训练并获取指标
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)

            if args.weight_d > 0:
                loss += reg_loss(net)

            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_index % 100 == 99:
                print(
                    f'[Epoch {epoch}, Batch {batch_index + 1}] Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0

        # 计算训练指标
        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # 验证并获取指标
        valid_acc, valid_loss, fs_valid, class_predict = eval_training(
            valid_loader,
            net,
            samples_per_cls=samples_per_cls,
            cb_config=cb_config,
            device=device,
            epoch=epoch
        )
        
        # 在验证函数中添加模型预测分布监控
        if epoch % 5 == 0 or epoch == args.epoch - 1:  # 每5个epoch或最后一个epoch检查一次
            # 将整数列表转换为张量
            all_preds = torch.tensor(class_predict)
            
            # 统计每个类别的预测数量
            pred_counts = torch.bincount(all_preds, minlength=args.num_classes)
            
            print("\n检查模型预测分布:")
            for i in range(args.num_classes):
                count = pred_counts[i].item()
                percentage = count / len(all_preds) * 100
                print(f"类别 {i} 的预测数量: {count} ({percentage:.2f}%)")
            
            # 检查是否有类别预测数量过少（可能出现类别崩塌）
            min_percentage = 0.05 * 100 / args.num_classes  # 期望每个类别至少有总预测的5%/类别数
            collapsed_classes = [i for i in range(args.num_classes) if (pred_counts[i] / len(all_preds) * 100) < min_percentage]
            
            if collapsed_classes:
                print("警告: 模型对某些类别的预测数量过少，可能出现类别崩塌问题。")
                print("建议: 1) 降低gamma值; 2) 降低beta值; 3) 尝试使用混合损失函数。")

        # 记录所有指标
        Train_Accuracy.append(float(train_acc))
        Train_Loss.append(float(train_loss))
        Valid_Accuracy.append(float(valid_acc))
        Valid_Loss.append(float(valid_loss))
        f1_s.append(float(fs_valid))
        
        # 在每个epoch结束后更新学习率
        train_scheduler.step()

        # 打印当前epoch的训练情况
        print(
            'Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | F1: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, fs_valid
            ))

        # 保存最佳模型
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)

        # 早停检查 - 基于验证准确率
        early_stopping(valid_acc)  # 传入验证准确率
        if early_stopping.early_stop:
            print(f'Early stopping triggered after {epoch} epochs - Best validation accuracy: {best_val_acc:.4f}')
            break

    # 评估测试集
    net.eval()
    test_target = []
    test_predict = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            test_target.extend(labels.cpu().numpy())
            test_predict.extend(predicted.cpu().numpy())

    # 绘制混淆矩阵
    cm = confusion_matrix(test_target, test_predict)

    # 定义类别标签
    class_names = ['Standing', 'Running', 'Grazing', 'Trotting', 'Walking']

    # 1. 计数混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                cmap="Blues",
                linecolor='white',
                linewidths=2,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True,
                fmt="d",
                cbar=True,
                square=True,
                annot_kws={'size': 12, 'weight': 'bold'})

    plt.title("Confusion Matrix (Counts)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')

    # 设置刻度标签
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'Confusion_matrix_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 百分比混淆矩阵
    plt.figure(figsize=(10, 8))
    cm_percentage = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    sns.heatmap(cm_percentage,
                cmap="Blues",
                linecolor='white',
                linewidths=2,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True,
                fmt=".1f",
                cbar=True,
                square=True,
                annot_kws={'size': 12, 'weight': 'bold'})

    plt.title("Confusion Matrix (Percentages)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')

    # 设置刻度标签
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'Confusion_matrix_percentages.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 绘制PR曲线和ROC曲线
    class_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

    # PR曲线
    plt.figure(figsize=(10, 8))
    for i in range(args.num_classes):
        y_true_binary = (np.array(test_target) == i).astype(int)
        y_score_binary = np.zeros((len(test_predict), args.num_classes))
        for j, pred in enumerate(test_predict):
            y_score_binary[j, pred] = 1

        precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary[:, i])
        avg_precision = average_precision_score(y_true_binary, y_score_binary[:, i])

        plt.plot(recall, precision,
                 color=class_colors[i],
                 label=f'{class_names[i]} (AP={avg_precision:.2f})',
                 linewidth=2)

    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=10)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC曲线
    plt.figure(figsize=(10, 8))
    for i in range(args.num_classes):
        y_true_binary = (np.array(test_target) == i).astype(int)
        y_score_binary = np.zeros((len(test_predict), args.num_classes))
        for j, pred in enumerate(test_predict):
            y_score_binary[j, pred] = 1

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr,
                 color=class_colors[i],
                 label=f'{class_names[i]} (AUC={roc_auc:.2f})',
                 linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('ROC Curves', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 绘制F1曲线
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(f1_s) + 1), f1_s, 'b-', label='F1 Score', linewidth=2)

    # 标记最佳F1分数
    best_f1_epoch = np.argmax(f1_s) + 1
    best_f1 = max(f1_s)
    plt.scatter(best_f1_epoch, best_f1, color='r', s=100, zorder=5)
    plt.annotate(f'Best: {best_f1:.4f}',
                 (best_f1_epoch, best_f1),
                 xytext=(10, 10),
                 textcoords='offset points')

    plt.title('F1 Score During Training', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(Train_Loss) + 1), Train_Loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, len(Valid_Loss) + 1), Valid_Loss, 'r-', label='Validation Loss', linewidth=2)

    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 绘制训练和验证准确率曲线
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(Train_Accuracy) + 1), Train_Accuracy, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(range(1, len(Valid_Accuracy) + 1), Valid_Accuracy, 'r-', label='Validation Accuracy', linewidth=2)

    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存分类报告
    classification_report_path = os.path.join(logs_dir, 'classification_report.txt')
    with open(classification_report_path, 'w') as f:
        # 添加标题和时间戳
        f.write('=' * 80 + '\n')
        f.write(f'分类报告 - 模型: {args.net} - 时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 80 + '\n\n')

        # 添加模型配置摘要
        f.write('模型配置:\n')
        f.write('-' * 50 + '\n')
        f.write(f'批量大小: {args.b}\n')
        f.write(f'学习率: {args.lr}\n')
        f.write(f'训练轮次: {args.epoch}\n')
        f.write(f'随机种子: {args.seed}\n')
        f.write(f'权重衰减: {args.weight_d}\n')
        f.write(f'Beta参数: {args.beta}\n')
        f.write(f'Gamma参数: {args.gamma}\n')
        f.write(f'类别数量: {args.num_classes}\n\n')

        # 添加类别分布信息
        f.write('类别分布:\n')
        f.write('-' * 50 + '\n')
        class_names = ['Standing', 'Running', 'Grazing', 'Trotting', 'Other']
        class_counts = np.bincount(test_target, minlength=args.num_classes)
        for i, (name, count) in enumerate(zip(class_names, class_counts)):
            f.write(f'类别 {i} ({name}): {count} 样本\n')
        f.write('\n')

        # 添加详细分类报告
        f.write('详细分类指标:\n')
        f.write('-' * 50 + '\n')
        # 使用类别名称替换数字标签
        report = classification_report(
            test_target, test_predict,
            target_names=class_names,
            zero_division=0
        )
        f.write(report + '\n')

        # 添加混淆矩阵
        f.write('混淆矩阵:\n')
        f.write('-' * 50 + '\n')
        matrix = confusion_matrix(test_target, test_predict)

        # 打印带有类别名称的混淆矩阵
        header = '真实\\预测'.ljust(15) + ''.join([name[:10].ljust(10) for name in class_names])
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')

        for i, row in enumerate(matrix):
            row_str = class_names[i][:10].ljust(15) + ''.join([str(x).ljust(10) for x in row])
            f.write(row_str + '\n')
        f.write('\n')

        # 添加性能摘要
        f.write('性能摘要:\n')
        f.write('-' * 50 + '\n')
        accuracy = np.mean(np.array(test_target) == np.array(test_predict))
        f.write(f'准确率: {accuracy:.4f}\n')

        # 计算每个类别的F1分数
        f1_per_class = f1_score(test_target, test_predict, average=None, zero_division=0)
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            f.write(f'{name} F1分数: {f1:.4f}\n')

        # 计算宏平均和加权平均F1
        macro_f1 = f1_score(test_target, test_predict, average='macro', zero_division=0)
        weighted_f1 = f1_score(test_target, test_predict, average='weighted', zero_division=0)
        f.write(f'宏平均F1: {macro_f1:.4f}\n')
        f.write(f'加权平均F1: {weighted_f1:.4f}\n')

        # 计算Cohen's Kappa
        kappa = cohen_kappa_score(test_target, test_predict)
        f.write(f'Cohen\'s Kappa: {kappa:.4f}\n\n')

        # 在文件关闭前打印GPU信息
        if args.gpu and torch.cuda.is_available():
            f.write('\nGPU信息:\n')
            f.write('-' * 50 + '\n')
            f.write(torch.cuda.memory_summary())