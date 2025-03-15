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
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc

class CBLossConfig:
    def __init__(self, num_classes, loss_type, beta, gamma):
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls, device):
    start = time.time()
    network.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        try:
            outputs = network(images)
            loss_type = "focal"
            
            # 使用动态权重的CB_loss
            loss = CB_loss(
                labels=labels,
                logits=outputs,
                samples_per_cls=samples_per_cls,
                no_of_classes=args.num_classes,
                loss_type=loss_type,
                beta=args.beta,
                gamma=args.gamma
            )
            
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
        
        # 使用CB_loss计算验证损失
        loss = CB_loss(
            labels=labels,
            logits=outputs,
            samples_per_cls=samples_per_cls,
            no_of_classes=args.num_classes,
            loss_type="focal",
            beta=cb_config.beta,
            gamma=cb_config.gamma
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
    return accuracy.cpu(), avg_loss, f1

        

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
            'loss_function': 'CB_loss with focal',
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST', help='net type')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoches')
    parser.add_argument('--seed',type=int, default=42, help='seed')
    parser.add_argument('--gamma', type=float, default=3.0, help='the gamma of focal loss')
    parser.add_argument('--beta', type=float, default=0.9999, help='the beta of class balanced loss')
    parser.add_argument('--weight_d', type=float, default=0.01, help='weight decay for regularization')
    parser.add_argument('--save_path', type=str, default='experiments/default_run',
                       help='path for saving all outputs (checkpoints, logs, etc)')
    parser.add_argument('--data_path',type=str, default='C:\\Users\\10025\\Desktop\\0000PatchTST-TFC-main\\0000PatchTST-TFC-main\\CMI-Net\\data\\new_goat_25hz_3axis.pt', help='saved path of input data')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes in the dataset')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm for gradient clipping')
    parser.add_argument('--num_workers', type=int, default=8 if platform.system() != "Windows" else 0, help='number of workers for data loading')
    parser.add_argument('--lr_decay', type=str, default='cosine',
                       help='learning rate decay type: cosine/step/linear (default: cosine)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    
    args = parser.parse_args()

    # 修改保存路径逻辑
    # 创建保存目录
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    # 创建时间戳子目录
    timestamp = settings.TIME_NOW
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

    device = torch.device("cuda:0" if args.gpu > 0 and torch.cuda.is_available() else "cpu")

    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpu = 0
        torch.manual_seed(args.seed)
    
    # 修改工作进程数设置部分
    sysstr = platform.system()
    num_workers = args.num_workers
        
    pathway = args.data_path
    if sysstr=='Linux': 
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
    
    # 创建模型
    net = get_network(args).to(device)
    
    # 打印模型结构详情
    print("\n" + "="*50)
    print("模型结构详情：")
    print(net)

    # 特别检查输出层
    print("\n输出层信息:")
    def get_output_dim(model):
        """获取模型输出维度的辅助函数"""
        if hasattr(model, 'fc'):
            if isinstance(model.fc, nn.Linear):
                return model.fc.out_features
            elif isinstance(model.fc, nn.Sequential):
                # 获取Sequential中的最后一个Linear层
                for layer in reversed(model.fc):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                return model.classifier.out_features
            elif isinstance(model.classifier, nn.Sequential):
                # 获取Sequential中的最后一个Linear层
                for layer in reversed(model.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
        
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
    
    print("="*50 + "\n")
    
    # 将 weight_train 移动到正确的设备上
    if isinstance(weight_train, torch.Tensor):
        weight_train = weight_train.to(device)
    
    # 确保 number_train 也在正确的设备上
    if isinstance(number_train, torch.Tensor):
        number_train = number_train.to(device)
    
    if args.weight_d > 0:
        reg_loss=Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")
    
    # 修改优化器配置
    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_d,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 修改学习率调度器
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
        else:  # linear decay
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return epoch / args.warmup_epochs
                else:
                    progress = (epoch - args.warmup_epochs) / (args.epoch - args.warmup_epochs)
                    return 1.0 - progress

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_scheduler = get_lr_scheduler(optimizer, args)

    # 更新梯度裁剪阈值
    max_grad_norm = args.max_grad_norm  # 使用参数化的梯度裁剪阈值

    best_acc = 0
    Train_Loss = []
    Train_Accuracy = []
    Valid_Loss = []
    Valid_Accuracy = []
    f1_s = []
    best_epoch = 0
    best_weights_path = checkpoint_path_pth.format(net=args.net, type='best')
    
    # 早停机制相关变量
    early_stopping_counter = 0
    best_valid_loss = float('inf')
    
    # 在训练开始前计算每个类别的样本数
    def get_class_counts(train_loader):
        class_counts = torch.zeros(args.num_classes)  # 使用参数化的类别数
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        return class_counts

    # 在训练主循环前计算
    samples_per_cls = get_class_counts(train_loader)

    print("Class distribution:")
    for i, count in enumerate(samples_per_cls):
        print(f"Class {i}: {count} samples")

    # 开始正式训练循环...
    for epoch in range(1, args.epoch + 1):
        # 训练并获取指标
        net.train()  # 确保模型在训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                outputs = net(images)
                loss = CB_loss(
                    labels=labels,
                    logits=outputs,
                    samples_per_cls=samples_per_cls,
                    no_of_classes=args.num_classes,
                    loss_type="focal",
                    beta=args.beta,
                    gamma=args.gamma
                )
            
            loss.backward()
            

            
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
        # 计算训练指标
        train_acc = correct / total  # 不乘以100，保持小数形式
        train_loss = running_loss / len(train_loader)
        
        # 更新学习率
        train_scheduler.step()
        
        # 验证并获取指标
        valid_acc, valid_loss, fs_valid = eval_training(
            valid_loader, 
            net, 
            samples_per_cls=samples_per_cls,
            cb_config=cb_config,
            device=device,
            epoch=epoch
        )
        
        # 记录所有指标 - 确保都是在CPU上
        Train_Accuracy.append(float(train_acc))  # 转换为Python float
        Train_Loss.append(float(train_loss))
        Valid_Accuracy.append(float(valid_acc))  # 转换为Python float
        Valid_Loss.append(float(valid_loss))
        f1_s.append(float(fs_valid))
        
        # 打印当前epoch的训练情况
        print('Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | F1: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, fs_valid
        ))
        
        # 保存最佳模型
        if epoch > settings.MILESTONES[0] and best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)
            
        # 早停机制
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
            if early_stopping_counter >= args.patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
            
    print('Best epoch: {} with accuracy: {:.4f}'.format(best_epoch, best_acc))


    #plot accuracy varying over time
    font_1 = {'weight' : 'normal', 'size'   : 20}
    fig1=plt.figure(figsize=(12,9))
    plt.title('Accuracy',font_1)
    index_train = list(range(1,len(Train_Accuracy)+1))
    plt.plot(index_train,Train_Accuracy,color='#FF9999',label='train_accuracy', linewidth=2)
    plt.plot(index_train,Valid_Accuracy,color='#66B2FF',label='valid_accuracy', linewidth=2)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Accuracy',font_1)
    
    acc_figuresavedpath = os.path.join(plots_dir, 'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    #plot loss varying over time
    fig2=plt.figure(figsize=(12,9))
    plt.title('Loss',font_1)
    index_valid = list(range(1,len(Valid_Loss)+1))
    plt.plot(index_valid,Train_Loss,color='#99FF99', label='train_loss', linewidth=2)
    plt.plot(index_valid,Valid_Loss,color='#FFCC99', label='valid_loss', linewidth=2)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    loss_figuresavedpath = os.path.join(plots_dir, 'Loss_curve.png')
    plt.savefig(loss_figuresavedpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    #plot f1 score varying over time
    fig3=plt.figure(figsize=(12,9))
    plt.title('F1-score',font_1)
    index_fs = list(range(1,len(f1_s)+1))
    plt.plot(index_fs,f1_s,color='#FF99CC', linewidth=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('F1-score',font_1)

    fs_figuresavedpath = os.path.join(plots_dir, 'F1-score.png')
    plt.savefig(fs_figuresavedpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存训练结果到txt文件
    out_txtsavedpath = os.path.join(logs_dir, 'output.txt')
    f = open(out_txtsavedpath, 'w+')
    
    # 保存实验配置信息
    print('Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Data path: {}, Saved path: {}'.format(
        args.seed, args.epoch, args.b, args.lr, args.weight_d, args.gpu, args.data_path, checkpoint_path),
        file=f)
    
    print('index: {}; maximum value of validation accuracy: {}.'.format(Valid_Accuracy.index(max(Valid_Accuracy))+1, max(Valid_Accuracy)), file=f)
    print('index: {}; maximum value of validation f1-score: {}.'.format(f1_s.index(max(f1_s))+1, max(f1_s)), file=f)
    print('--------------------------------------------------')
    print('Validation accuracy: {}'.format(Valid_Accuracy), file=f)
    print('Validation F1-score: {}'.format(f1_s), file=f)
    
    # 保存CSV结果
    csv_path = os.path.join(logs_dir, 'results.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w+') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow(['index','accuracy','f1-score','precision','recall','kappa','time_consumed'])

    ######load the best trained model and test testing data  ，测试函数，推理
    best_net = get_network(args)
    best_net.load_state_dict(torch.load(best_weights_path, map_location=device))
    best_net = best_net.to(device)
    
    total_num_paras, trainable_num_paras = get_parameter_number(best_net)
    print('The total number of network parameters = {}'.format(total_num_paras), file=f)
    print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)
    
    best_net.eval()
    number = 0
    correct_test = 0.0
    test_target =[]
    test_predict = []
    
    # 定义行为标签
    behavior_labels = ['standing', 'running', 'grazing', 'trotting', 'walking']
    
    with torch.no_grad():
        start = time.time()
        
        for n_iter, (image, labels) in enumerate(test_loader):
            image = image.to(device)
            labels = labels.to(device)

            output = best_net(image)
            output = torch.softmax(output, dim= 1)
            preds = torch.argmax(output, dim =1)
            correct_test += preds.eq(labels).sum()
            
            labels = labels.cpu()
            preds = preds.cpu()
        
            test_target.extend(labels.numpy().tolist())
            test_predict.extend(preds.numpy().tolist())
        
            number +=1
        
        finish = time.time()
        accuracy_test = correct_test.float() / len(test_loader.dataset)
        
        # 计算各项指标
        fs_test = f1_score(test_target, test_predict, average='macro')
        kappa_value = cohen_kappa_score(test_target, test_predict)
        precision_test = precision_score(test_target, test_predict, average='macro', zero_division=0)
        recall_test = recall_score(test_target, test_predict, average='macro', zero_division=0)
        
        # 打印测试集评估结果
        print('\nEvaluating Network.....')
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
            0.0,  # 由于我们在测试阶段没有计算损失，这里填0
            accuracy_test,
            finish - start
        ))
        
        # 打印分类报告
        print('Classification Report:')
        print(classification_report(test_target, test_predict, zero_division=0))
        
        # 打印混淆矩阵
        print('\nConfusion Matrix - Test Results:')
        matrix = confusion_matrix(test_target, test_predict)
        
        # 打印列索引
        n_classes = len(matrix)
        print('\t' + '\t'.join([str(i) for i in range(n_classes)]))
        print('-' * (8 * n_classes))
        
        # 打印每一行
        for i in range(n_classes):
            row = [str(x) for x in matrix[i]]
            print(f'{i}\t' + '\t'.join(row))
        print()
        
        # 打印总体训练结果
        print('\n' + '='*50)
        print('Training Summary')
        print('='*50)
        print(f'Epoch: {best_epoch}, Average loss: {0.0:.4f}, Accuracy: {accuracy_test:.4f}, Time consumed: {finish - start:.2f}s')
        print(f'F1 Score: {fs_test:.4f}')
        print(f'Precision: {precision_test:.4f}')
        print(f'Recall: {recall_test:.4f}')
        print(f'Kappa Score: {kappa_value:.4f}')
        print(f'Total samples: {len(test_loader.dataset)}')
        print(f'Total correct: {int(correct_test)}')
        print(f'Average time per sample: {(finish - start)/len(test_loader.dataset):.4f}s')
        print('='*50 + '\n')
        
        # 绘制混淆矩阵
        def show_confusion_matrix(validations, predictions):
            matrix = confusion_matrix(validations, predictions)
            percentages = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            
            # 第一个混淆矩阵：显示数量
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix,
                  cmap="Blues",
                  linecolor='white',
                  linewidths=2,
                  xticklabels=behavior_labels,
                  yticklabels=behavior_labels,
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
            cm_counts_path = os.path.join(plots_dir, 'Confusion_matrix_counts.png')
            plt.savefig(cm_counts_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # 第二个混淆矩阵：显示百分比
            plt.figure(figsize=(10, 8))
            sns.heatmap(percentages,
                  cmap="Blues",
                  linecolor='white',
                  linewidths=2,
                  xticklabels=behavior_labels,
                  yticklabels=behavior_labels,
                  annot=True,
                  fmt=".1%",
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
            cm_percentages_path = os.path.join(plots_dir, 'Confusion_matrix_percentages.png')
            plt.savefig(cm_percentages_path, bbox_inches='tight', dpi=300)
            plt.close()

        # 绘制PR曲线和ROC曲线
        def plot_pr_roc_curves(y_true, y_score):
            # 为每个类别定义不同的颜色
            class_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
            
            # 绘制PR曲线
            plt.figure(figsize=(10, 8))
            for i in range(len(behavior_labels)):
                y_true_binary = (np.array(y_true) == i).astype(int)
                y_score_binary = y_score[:, i]
                
                precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
                avg_precision = average_precision_score(y_true_binary, y_score_binary)
                
                plt.plot(recall, precision, color=class_colors[i], 
                        label=f'{behavior_labels[i]} (AP={avg_precision:.2f})', 
                        linewidth=2)
            
            plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Recall', fontsize=14, fontweight='bold')
            plt.ylabel('Precision', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower left', fontsize=10)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            plt.tight_layout()
            pr_curve_path = os.path.join(plots_dir, 'PR_curve.png')
            plt.savefig(pr_curve_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # 绘制ROC曲线
            plt.figure(figsize=(10, 8))
            for i in range(len(behavior_labels)):
                y_true_binary = (np.array(y_true) == i).astype(int)
                y_score_binary = y_score[:, i]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=class_colors[i], 
                        label=f'{behavior_labels[i]} (AUC={roc_auc:.2f})', 
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
            roc_curve_path = os.path.join(plots_dir, 'ROC_curve.png')
            plt.savefig(roc_curve_path, bbox_inches='tight', dpi=300)
            plt.close()

        # 绘制混淆矩阵
        show_confusion_matrix(test_target, test_predict)
        
        # 计算并绘制PR曲线和ROC曲线
        test_probabilities = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = best_net(images)
                probs = torch.softmax(outputs, dim=1)
                test_probabilities.append(probs.cpu().numpy())
        
        test_probabilities = np.vstack(test_probabilities)
        plot_pr_roc_curves(test_target, test_probabilities)
        
        # 创建训练配置字典
        train_config = {
            'best_accuracy': best_acc,
            'best_epoch': best_epoch,
            'final_metrics': {
                'accuracy': float(accuracy_test),
                'f1_score': float(fs_test),
                'precision': float(precision_test),
                'recall': float(recall_test),
                'kappa': float(kappa_value)
            }
        }
        
        # 保存模型配置
        save_model_config(best_net, args, train_config, checkpoint_path)

    if args.gpu and torch.cuda.is_available():
        print('GPU INFO.....', file=f)
        print(torch.cuda.memory_summary(), end='', file=f)

    