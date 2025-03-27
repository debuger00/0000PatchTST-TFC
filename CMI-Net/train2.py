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
from sklearn.manifold import TSNE  # 添加TSNE导入
from sklearn.neighbors import KNeighborsClassifier  # 添加KNN导入

# 添加Lion优化器实现
class Lion(optim.Optimizer):
    """Lion optimizer - Energy-Efficient Adaptive Optimization.
    
    Paper: https://arxiv.org/abs/2302.06675
    Better than Adam/AdamW for imbalanced datasets and converges faster.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-4)
        betas: coefficients used for computing running averages (default: (0.9, 0.99))
        weight_decay: weight decay coefficient (default: 0.0)
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update weights
                update = exp_avg.sign()
                p.data.add_(update, alpha=-group['lr'])
                
        return loss


class CBLossConfig:
    def __init__(self, num_classes, loss_type, beta, gamma):
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False, monitor='f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor  # 'f1' for F1 score

    def __call__(self, score):
        # For F1 score, higher is better
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


def train(net, train_loader, valid_loader, criterion, optimizer, scheduler, args, device, samples_per_cls):
    net.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
        
        # 
        if batch_idx % 500 == 0 and batch_idx > 0:
            print(f'Epoch 进度: Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 计算训练指标
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # 返回训练准确率和损失
    return net, epoch_accuracy, epoch_loss


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
            loss = cb_ce_mixed_loss(
                labels, outputs, samples_per_cls, args.num_classes, 
                'focal', args.beta, args.gamma, args.ce_weight
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
    print()  

    accuracy = correct.float() / len(valid_loader.dataset)
    avg_loss = valid_loss / n
    f1 = f1_score(class_target, class_predict, average='macro', zero_division=0)

    # 确保返回的是CPU张量
    return accuracy.cpu(), avg_loss, f1, class_predict


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def save_model_config(model, args, train_config, save_path):
    config = {
        'model_name': args.net,
        'input_size': train_config['input_size'],
        'num_classes': args.num_classes,
        'batch_size': args.b,
        'learning_rate': args.lr,
        'epochs': args.epoch,
        'optimizer': args.optimizer,
        'loss_function': 'CB-CE Mixed Loss' if args.use_mixed_loss == 1 else 'CB Loss with focal',
        'ce_weight': args.ce_weight if args.use_mixed_loss == 1 else 0,
        'beta': args.beta,
        'gamma': args.gamma,
        'weight_decay': args.weight_d,
        'regularization_type': args.reg_type,
        'device': str(args.device),
        'model_architecture': str(model)
    }
    
    # 保存配置到YAML文件
    config_path = os.path.join(save_path, 'model_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# 添加混合损失函数
def cb_ce_mixed_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, ce_weight=0.3):
    """
    结合CB Loss和CE Loss的混合损失函数。
    
    参数:
        labels (torch.Tensor): 形状为[batch]的整数张量，包含每个样本的类别标签。
        logits (torch.Tensor): 形状为[batch, no_of_classes]的浮点张量，包含模型的原始输出分数。
        samples_per_cls (list/torch.Tensor): 大小为[no_of_classes]的列表，包含每个类别的样本总数。
        no_of_classes (int): 类别总数。
        loss_type (str): CB Loss使用的基础损失函数类型，可选"sigmoid"、"focal"或"softmax"。
        beta (float): 类别平衡的超参数，用于计算有效样本数。
        gamma (float): Focal Loss的聚焦参数，仅在loss_type="focal"时使用。
        ce_weight (float): CE Loss在混合损失中的权重，范围[0,1]，默认0.3。
        
    返回:
        torch.Tensor: 标量张量，表示计算得到的混合损失值。
    """
    device = logits.device
    
    # 计算CB Loss
    cb_loss_val = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)
    
    # 计算CE Loss
    ce_loss_val = F.cross_entropy(logits, labels)
    
    # 组合两种损失
    total_loss = (1.0 - ce_weight) * cb_loss_val + ce_weight * ce_loss_val
    
    return total_loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST_Classification', help='net type')
    parser.add_argument('--gpu', type=int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=150, help='total training epoches')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--gamma', type=float, default=0.5, help='the gamma of focal loss')
    parser.add_argument('--beta', type=float, default=0.99, help='the beta of class balanced loss')
    parser.add_argument('--weight_d', type=float, default=0.001, help='weight decay for regularization')
    parser.add_argument('--reg_type', type=str, default='L2', choices=['L1', 'L2', 'none'], help='regularization type: L1, L2 or none')
    parser.add_argument('--save_path', type=str, default='experiments/default_run',
                        help='path for saving all outputs (checkpoints, logs, etc)') 
    parser.add_argument('--data_path', type=str,
                        default='./data/new_goat_25hz_3axis.pt', 
                        help='saved path of input data')
    parser.add_argument('--patience', type=int, default=25, help='patience for early stopping')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes in the dataset')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm for gradient clipping')
    parser.add_argument('--num_workers', type=int, default=8 if platform.system() != "Windows" else 0,
                        help='number of workers for data loading')
    parser.add_argument('--lr_decay', type=str, default='cyclic',
                        choices=['cosine', 'step', 'cyclic', 'onecycle', 'plateau'],
                        help='learning rate decay type: cosine/step/cyclic/onecycle/plateau (default: cosine)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'lion'],
                        help='optimizer to use: adamw/adam/sgd/lion (default: adamw)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--cycle_epochs', type=int, default=10, help='number of epochs per cycle for cyclic lr')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',help='device to use')
    parser.add_argument('--use_mixed_loss', type=int, default=1, help='loss type: 0 for CB Loss only, 1 for CB-CE mixed loss')
    parser.add_argument('--ce_weight', type=float, default=0.2, help='weight of CE Loss in CB-CE mixed loss')
    parser.add_argument('--weight_smooth', type=int, default=1, help='whether to apply weight smoothing: 0 for no smoothing, 1 for log smoothing')
    parser.add_argument('--focal_alpha', type=float, default=None, help='alpha parameter for focal loss, if None, use class weights')
    # T-SNE可视化相关参数
    parser.add_argument('--tsne_perplexity', type=float, default=30.0, help='T-SNE困惑度参数，影响局部结构保留程度')
    parser.add_argument('--tsne_n_iter', type=int, default=1000, help='T-SNE迭代次数，影响结果质量和运行时间')
    parser.add_argument('--tsne_learning_rate', type=float, default=200.0, help='T-SNE学习率，影响收敛速度')
    parser.add_argument('--visualize_features', type=int, default=0, help='是否进行特征可视化: 0为否，1为是')

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
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            print(f"使用余弦学习率调整，预热轮数: {args.warmup_epochs} epochs")
            
        elif args.lr_decay == 'step':
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return epoch / args.warmup_epochs
                else:
                    # 每30个epoch衰减为原来的0.1
                    return 0.1 ** (epoch // 30)
                    
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            print(f"使用阶梯式学习率调整，预热轮数: {args.warmup_epochs} epochs")
            
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
            
        elif args.lr_decay == 'onecycle':
            # OneCycleLR - 高效防止过拟合，特别适合不平衡数据集
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr * 10,  # 最大学习率为初始学习率的10倍
                total_steps=args.epoch,
                pct_start=0.3,  # 上升阶段占总步数的30%
                anneal_strategy='cos',  # 使用余弦衰减
                div_factor=25.0,  # 初始学习率 = max_lr / div_factor
                final_div_factor=10000.0,  # 最终学习率 = max_lr / final_div_factor
                three_phase=False  # 使用两阶段策略
            )
            print(f"使用OneCycle学习率调整，总轮数: {args.epoch}，最大学习率: {args.lr * 10}")
            
        elif args.lr_decay == 'plateau':
            # 自适应调整学习率 - 验证损失停滞时降低学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',  # 因为我们监控验证准确率
                factor=0.5,  # 学习率衰减因子
                patience=5,   # 等待多少个epoch
                verbose=True,
                threshold=1e-4,
                min_lr=args.min_lr
            )
            print(f"使用ReduceLROnPlateau学习率调整，监控指标: 验证准确率")
            return scheduler  # 特殊情况，需要在epoch结束时传入验证集性能指标
        
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

    # 根据参数选择正则化类型
    if args.weight_d > 0 and args.reg_type != 'none':
        if args.reg_type == 'L1':
            reg_loss = Regularization(net, args.weight_d, p=1)
            print(f"使用 L1 正则化，权重衰减值: {args.weight_d}")
        else:  # L2
            reg_loss = Regularization(net, args.weight_d, p=2)
            print(f"使用 L2 正则化，权重衰减值: {args.weight_d}")
    else:
        print("不使用显式正则化")
        reg_loss = None

    # 修改优化器参数 - 根据正则化类型设置weight_decay
    if args.reg_type == 'none':
        # 如果不使用显式正则化，仍然可以使用优化器内置的权重衰减
        optimizer_weight_decay = args.weight_d
    else:
        # 如果使用显式正则化，避免重复正则化，将优化器的权重衰减设为0
        optimizer_weight_decay = 0
    
    # 根据参数选择优化器
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=optimizer_weight_decay)
        print(f"使用AdamW优化器，学习率: {args.lr}, 权重衰减: {optimizer_weight_decay}")
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=optimizer_weight_decay)
        print(f"使用Adam优化器，学习率: {args.lr}, 权重衰减: {optimizer_weight_decay}")
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=optimizer_weight_decay)
        print(f"使用SGD优化器，学习率: {args.lr}, 动量: 0.9, 权重衰减: {optimizer_weight_decay}")
    elif args.optimizer == 'lion':
        optimizer = Lion(net.parameters(), lr=args.lr, weight_decay=optimizer_weight_decay)
        print(f"使用Lion优化器，学习率: {args.lr}, 权重衰减: {optimizer_weight_decay}")
    else:
        # 默认使用AdamW
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=optimizer_weight_decay)
        print(f"使用AdamW优化器，学习率: {args.lr}, 权重衰减: {optimizer_weight_decay}")
        
    if optimizer_weight_decay > 0:
        print(f"优化器使用内置权重衰减: {optimizer_weight_decay}")
        
    train_scheduler = get_lr_scheduler(optimizer, args)

    # 修改早停策略 - 基于F1分数
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
        verbose=True,
        monitor='f1'  # 明确指定监控F1分数
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
        print(f"\n使用混合损失函数 (CB Loss + CE Loss)")
        print(f"CE Loss权重: {args.ce_weight}")
        # 创建CB-CE混合损失函数
        criterion = lambda outputs, labels: cb_ce_mixed_loss(
            labels, outputs, samples_per_cls, args.num_classes, 
            'focal', args.beta, args.gamma, args.ce_weight
        )
    else:
        print(f"\n使用CB Loss ({cb_config.loss_type})")
        print(f"Beta: {args.beta}, Gamma: {args.gamma}")
        # 创建CB Loss函数
        criterion = lambda outputs, labels: CB_loss(
            labels, outputs, samples_per_cls, args.num_classes, 
            cb_config.loss_type, args.beta, args.gamma
        )

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

    print("=" * 50)
    print(f"[损失函数配置] Beta: {args.beta}, Gamma: {args.gamma}")
    if args.use_mixed_loss == 1:
        print(f"使用混合损失函数，CE Loss权重: {args.ce_weight}")
    print("=" * 50)
    
    # 开始训练循环
    for epoch in range(1, args.epoch + 1):
        # 添加分隔符使输出更清晰
        print(f"开始训练第 {epoch}/{args.epoch} 轮")
        print("="*70)
        
        # 训练模型 - 不再在train函数中进行验证
        net, train_acc, train_loss = train(
            net,
            train_loader,
            None,  # 不传入验证集
            criterion,
            optimizer,
            train_scheduler,
            args,
            device,
            samples_per_cls
        )

        # 在主循环中进行验证
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
            # 收集所有预测 - 将整数列表转换为张量
            all_preds = torch.tensor(class_predict, device=device)
            
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
            
        # 记录所有指标
        Train_Accuracy.append(float(train_acc))
        Train_Loss.append(float(train_loss))
        Valid_Accuracy.append(float(valid_acc))
        Valid_Loss.append(float(valid_loss))
        f1_s.append(float(fs_valid))
        
        # 在每个epoch结束后更新学习率
        if args.lr_decay == 'plateau':
            # ReduceLROnPlateau需要监控指标
            train_scheduler.step(valid_acc)
        else:
            # 其他调度器只需要step()
            train_scheduler.step()

        # 打印当前epoch的训练情况
        print(
            'Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | F1: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, fs_valid
            ))

        # 保存最佳模型
        if fs_valid > best_val_acc:
            best_val_acc = fs_valid
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)
            print(f"模型保存成功! 当前最佳验证集F1分数: {best_val_acc:.4f}")

        # 早停检查 - 基于F1分数
        early_stopping(fs_valid)  # 传入F1分数
        if early_stopping.early_stop:
            print(f'早停在第 {epoch} 轮触发 - 最佳F1分数: {best_val_acc:.4f}')
            break

        # 添加分隔符表示轮次结束
        print("-"*70)

    # 评估测试集
    net.eval()
    test_target = []
    test_predict = []
    test_features = []  # 用于存储特征向量
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 获取特征向量和预测
            # 使用我们之前添加的get_features方法
            if hasattr(net, 'get_features'):
                features, outputs = net.get_features(images)
                test_features.append(features.cpu().numpy())
            else:
                outputs = net(images)
                # 如果没有get_features方法，我们不收集特征向量
            
            _, predicted = outputs.max(1)
            test_target.extend(labels.cpu().numpy())
            test_predict.extend(predicted.cpu().numpy())
    
    # 如果没有收集到特征向量，跳过可视化部分
    if len(test_features) == 0:
        print("\n警告: 未收集到特征向量，无法进行特征可视化。请确保模型有get_features方法。")
    else:
        # 将所有批次的特征连接起来
        test_features = np.vstack(test_features)
        
        # 使用T-SNE降维和可视化
        if args.visualize_features == 1:
            print("\n正在使用T-SNE进行特征降维...")
            print(f"T-SNE参数: 困惑度={args.tsne_perplexity}, 迭代次数={args.tsne_n_iter}, 学习率={args.tsne_learning_rate}")
            
            # 计算合适的困惑度，避免值过大导致错误
            perplexity = min(args.tsne_perplexity, len(test_features) - 1)
            if perplexity != args.tsne_perplexity:
                print(f"警告: 困惑度参数过大，已自动调整为 {perplexity}")
            
            # 使用T-SNE降维
            tsne = TSNE(
                n_components=2,
                random_state=args.seed,
                perplexity=perplexity,
                n_iter=args.tsne_n_iter,
                learning_rate=args.tsne_learning_rate
            )
            tsne_results = tsne.fit_transform(test_features)
            
            # 绘制T-SNE可视化图
            plt.figure(figsize=(12, 10))
            
            # 定义类别标签和颜色
            class_names = ['Standing', 'Running', 'Grazing', 'Trotting', 'Walking']
            class_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            
            # 为每个类别绘制散点图
            for i, class_name in enumerate(class_names):
                # 找到属于当前类别的点
                indices = np.array(test_target) == i
                
                # 绘制散点图
                plt.scatter(
                    tsne_results[indices, 0],
                    tsne_results[indices, 1],
                    c=class_colors[i],
                    label=class_name,
                    alpha=0.7,
                    edgecolors='w',
                    s=100
                )
            
            # 添加标题和图例 - 使用英文替代中文
            plt.title('T-SNE Feature Visualization', fontsize=20, fontweight='bold', pad=20)
            plt.xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
            plt.ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
            plt.legend(fontsize=14, markerscale=1.5, loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加边框
            ax = plt.gca()
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"T-SNE visualization saved to {os.path.join(plots_dir, 'tsne_visualization.png')}")
            
            # 额外绘制带有决策边界的T-SNE图
            try:
                print("Generating T-SNE visualization with decision boundaries...")
                
                # 创建网格
                h = 0.1  # 网格步长
                x_min, x_max = tsne_results[:, 0].min() - 1, tsne_results[:, 0].max() + 1
                y_min, y_max = tsne_results[:, 1].min() - 1, tsne_results[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                
                # 使用KNN分类器拟合T-SNE结果
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(tsne_results, test_target)
                
                # 预测网格点的类别
                Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # 绘制决策边界
                plt.figure(figsize=(12, 10))
                plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
                
                # 绘制散点图
                for i, class_name in enumerate(class_names):
                    indices = np.array(test_target) == i
                    plt.scatter(
                        tsne_results[indices, 0],
                        tsne_results[indices, 1],
                        c=class_colors[i],
                        label=class_name,
                        alpha=0.8,
                        edgecolors='k',
                        s=80
                    )
                
                plt.title('T-SNE Feature Visualization with Decision Boundaries', fontsize=20, fontweight='bold', pad=20)
                plt.xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
                plt.ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
                plt.legend(fontsize=14, markerscale=1.5, loc='best')
                
                # 保存图像
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'tsne_with_boundaries.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"T-SNE visualization with decision boundaries saved to {os.path.join(plots_dir, 'tsne_with_boundaries.png')}")
            except Exception as e:
                print(f"Error generating decision boundaries: {str(e)}")
        
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
        class_names = ['Standing', 'Running', 'Grazing', 'Trotting', 'Walking']
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