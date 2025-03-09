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
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix

class CBLossConfig:
    def __init__(self, num_classes, loss_type, beta, gamma):
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

def train(train_loader, network, optimizer, epoch, loss_function, samples_per_cls):
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
            loss = CB_loss(labels, outputs, samples_per_cls, 5, loss_type, args.beta, args.gamma)
            
            if args.weight_d > 0:
                loss += reg_loss(network)
            
            loss.backward()
            
            # 计算梯度范数
            total_norm = 0.0
            for p in network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # 如果梯度过大，进行缩放
            if total_norm > 10.0:
                scale = 10.0 / total_norm
                for p in network.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(scale)
            
            optimizer.step()

            if batch_index % 100 == 0:  # 每100个批次打印一次梯度信息
                print(f"Epoch {epoch}, Batch {batch_index}, Gradient Norm: {total_norm:.4f}")

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
def eval_training(valid_loader, network, loss_function, samples_per_cls, cb_config, epoch=0):
    start = time.time()
    network.eval()

    n = 0
    valid_loss = 0.0
    correct = 0.0
    class_target = []
    class_predict = []

    # 在 GPU 上收集所有数据，减少频繁的 GPU 到 CPU 转换
    for (images, labels) in valid_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = network(images)
        loss = CB_loss(
            labels, 
            outputs, 
            samples_per_cls, 
            cb_config.num_classes, 
            cb_config.loss_type, 
            cb_config.beta, 
            cb_config.gamma
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='PatchTST', help='net type')
    parser.add_argument('--gpu', type = int, default=1, help='use gpu or not')  # 选择是否使用 GPU（1 表示使用 GPU，0 表示使用 CPU）。
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoches')
    parser.add_argument('--seed',type=int, default=10, help='seed')
    parser.add_argument('--gamma', type=float, default=5, help='the gamma of focal loss')
    parser.add_argument('--beta', type=float, default=0.999, help='the beta of class balanced loss')
    parser.add_argument('--weight_d', type=float, default=0.0000, help='weight decay for regularization')
    parser.add_argument('--save_path',type=str, default='setting0', help='saved path of each setting') #
    # parser.add_argument('--data_path',type=str, default='C:\\Users\\10025\\Desktop\\0000PatchTST-TFC-main\\0000PatchTST-TFC-main\\CMI-Net\\data\\new_goat_25hz_3axis.pt', help='saved path of input data')
    parser.add_argument('--data_path',type=str, default='/data1/wangyonghua/program/0000PatchTST-TFC/CMI-Net/data/new_goat_25hz_3axis.pt', help='saved path of input data')
    args = parser.parse_args()

    # 创建配置对象
    cb_config = CBLossConfig(
        num_classes=5,
        loss_type="focal",
        beta=args.beta,
        gamma=args.gamma
    )

    device = torch.device("cuda:0" if args.gpu > 0 and torch.cuda.is_available() else "cpu") # 条件运算符，如果 args.gpu > 0 并且 torch.cuda.is_available() 为 True，则使用 GPU，否则使用 CPU

    if args.gpu:
        torch.cuda.manual_seed(args.seed)# 设置 GPU 上的随机数种子，确保在 GPU 上的随机操作（如权重初始化等）也是可重复的
    else:
        torch.manual_seed(args.seed)#  设置 CPU 上的随机数种子，确保在 CPU 上执行的所有与随机性相关的操作都是可重复的
    
    net = get_network(args).to(device)   # get_network 在 utils.py  中 ，把模型搬运到device(GPU)中
    
    # 打印模型结构详情
    print("\n" + "="*50)
    print("模型结构详情：")
    print(net)  # 输出完整的模型结构

    # 特别检查输出层
    print("\n输出层信息:")
    if hasattr(net, 'classifier'):
        if isinstance(net.classifier, nn.Sequential):
            # 获取Sequential中的最后一个Linear层
            for layer in reversed(net.classifier):
                if isinstance(layer, nn.Linear):
                    print("输出层维度:", layer.out_features)
                    output_dim = layer.out_features
                    break
        else:
            print("输出层维度:", net.classifier.out_features)
            output_dim = net.classifier.out_features
    elif hasattr(net, 'fc'):
        print("输出层维度:", net.fc.out_features)
        output_dim = net.fc.out_features
    else:
        print("警告：无法直接获取输出层维度！")
        # 尝试通过模型的最后一层获取维度
        last_layer = None
        for module in net.modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        if last_layer is not None:
            print("通过最后一个Linear层获取维度:", last_layer.out_features)
            output_dim = last_layer.out_features
        else:
            raise AttributeError("无法找到模型的输出层！")
    print("="*50 + "\n")

    print(f"Model is on device: {next(net.parameters()).device}")
    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(
        args.epoch, args.b, args.lr, args.gpu, args.seed))

    sysstr = platform.system()
    if(sysstr =="Windows"):
        num_workers = 0
    else:
        num_workers = 8                        # 在windows上的进程是0， 在Linux的是8？ 在Windows 在多进程的数据加载时可能会遇到问题？？？？
        
    pathway = args.data_path                     # 默认Linux的问题
    if sysstr=='Linux': 
        pathway = args.data_path
    
    train_loader, weight_train, number_train = get_weighted_mydataloader(
        pathway, 
        data_id=0, 
        batch_size=args.b, 
        num_workers=num_workers, 
        shuffle=True
    )
    valid_loader = get_mydataloader(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=True)
    
    # 检查数据集信息
    print("\n" + "="*50)
    print("数据集信息检查：")
    
    # 检查类别数 - 使用更健壮的方式获取类别数
    try:
        if hasattr(train_loader.dataset, 'classes'):
            num_classes = len(train_loader.dataset.classes)
        else:
            # 如果没有classes属性，尝试从数据中推断类别数
            num_classes = len(torch.unique(torch.tensor([label for _, label in train_loader.dataset])))
        print(f"数据集的类别数: {num_classes}")
        
        # 对比模型输出层维度
        if hasattr(net, 'fc'):
            output_dim = net.fc.out_features
        elif hasattr(net, 'classifier'):
            output_dim = net.classifier.out_features
        else:
            raise AttributeError("无法找到模型的输出层！")
            
        print(f"模型输出层维度: {output_dim}")
        
        assert output_dim == num_classes, f"警告：模型输出层维度({output_dim})与数据类别数({num_classes})不匹配！"
        print("✓ 模型输出维度与数据类别数匹配")
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
    
    # 使用标签平滑的交叉熵损失
    loss_function_CE = nn.CrossEntropyLoss(
        weight=weight_train,
        label_smoothing=0.1  # 添加标签平滑
    )
    # 修改优化器配置
    optimizer = optim.AdamW(
        net.parameters(),
        lr=0.001,  # 降低初始学习率
        weight_decay=1e-4,  # 增加权重衰减
        betas=(0.9, 0.999),  # 使用默认动量参数
        eps=1e-8
    )

    # 使用带预热的学习率调度器
    num_epochs = args.epoch
    warmup_epochs = 5
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return epoch / warmup_epochs
        else:
            # 余弦退火
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 添加梯度裁剪
    max_grad_norm = 1.0  # 设置梯度裁剪阈值

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):               # 如果没 log 路径 创建log路径
        os.mkdir(settings.LOG_DIR)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):                  # 参数路径
        os.makedirs(checkpoint_path)
    checkpoint_path_pth = os.path.join(checkpoint_path, '{net}-{type}.pth')

    best_acc = 0.0
    Train_Loss = []
    Train_Accuracy = []
    Valid_Loss = []
    Valid_Accuracy = []
    f1_s = []
    best_epoch = 1
    best_weights_path = checkpoint_path_pth.format(net=args.net, type='best')
   
    # 在主训练循环前添加过拟合测试
    print("\n" + "="*50)
    print("开始过拟合测试")
    print("目的：验证模型是否有足够的容量学习数据")
    print("预期：在小数据集上损失应该快速下降，准确率接近100%")
    print("="*50)

    # 创建非常小的数据集用于过拟合测试
    small_dataset_size = 100  # 只使用100个样本
    indices = torch.randperm(len(train_loader.dataset))[:small_dataset_size]
    small_dataset = torch.utils.data.Subset(train_loader.dataset, indices)
    
    small_train_loader = DataLoader(
        small_dataset,
        batch_size=16,  # 使用更小的batch size
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 临时创建新的优化器用于过拟合测试
    overfit_optimizer = optim.Adam(
        net.parameters(),
        lr=0.001,  # 使用更小的学习率
        weight_decay=0
    )

    print("\n--- 过拟合测试：在小批量数据上训练 ---")
    print(f"使用数据集大小: {small_dataset_size}, Batch size: 16")
    
    for epoch in range(20):  # 增加训练轮数
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(small_train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            overfit_optimizer.zero_grad()
            
            with autocast():
                outputs = net(inputs)
                loss = F.cross_entropy(outputs, targets)  # 使用简单的交叉熵损失
            
            loss.backward()
            overfit_optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(small_train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"过拟合测试 Epoch {epoch}: 准确率 = {epoch_acc:.2f}%, 损失 = {epoch_loss:.4f}")
        
        # 如果准确率达到很高，提前停止
        if epoch_acc > 95:
            print(f"\n✓ 过拟合测试成功！模型在epoch {epoch}达到{epoch_acc:.2f}%的准确率")
            break
    
    print("\n" + "="*50)
    print("过拟合测试完成")
    print("即将开始正式训练...")
    print("="*50 + "\n")

    # 重新初始化模型和优化器
    net = get_network(args).to(device)
    
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
                loss = CB_loss(labels, outputs, number_train, 5, "focal", args.beta, args.gamma)
            
            loss.backward()
            
            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            if batch_index % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_index}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 计算训练指标
        train_acc = correct / total  # 不乘以100，保持小数形式
        train_loss = running_loss / len(train_loader)
        
        # 更新学习率
        train_scheduler.step()
        
        # 验证并获取指标
        valid_acc, valid_loss, fs_valid = eval_training(
            valid_loader, 
            net, 
            loss_function_CE, 
            samples_per_cls=number_train,  # 添加这个参数
            cb_config=cb_config,
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
            
    print('Best epoch: {} with accuracy: {:.4f}'.format(best_epoch, best_acc))


    #plot accuracy varying over time
    font_1 = {'weight' : 'normal', 'size'   : 20}
    fig1=plt.figure(figsize=(12,9))
    plt.title('Accuracy',font_1)
    index_train = list(range(1,len(Train_Accuracy)+1))
    plt.plot(index_train,Train_Accuracy,color='skyblue',label='train_accuracy')
    plt.plot(index_train,Valid_Accuracy,color='red',label='valid_accuracy')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Accuracy',font_1)
    
    acc_figuresavedpath = os.path.join(checkpoint_path,'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath)
    # plt.show()
    
    #plot loss varying over time
    fig2=plt.figure(figsize=(12,9))
    plt.title('Loss',font_1)
    index_valid = list(range(1,len(Valid_Loss)+1))
    plt.plot(index_valid,Train_Loss,color='skyblue', label='train_loss')
    plt.plot(index_valid,Valid_Loss,color='red', label='valid_loss')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    loss_figuresavedpath = os.path.join(checkpoint_path,'Loss_curve.png')
    plt.savefig(loss_figuresavedpath)
    # plt.show()
    
    #plot f1 score varying over time
    fig3=plt.figure(figsize=(12,9))
    plt.title('F1-score',font_1)
    index_fs = list(range(1,len(f1_s)+1))
    plt.plot(index_fs,f1_s,color='skyblue')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    fs_figuresavedpath = os.path.join(checkpoint_path,'F1-score.png')
    plt.savefig(fs_figuresavedpath)
    # plt.show()
    
    out_txtsavedpath = os.path.join(checkpoint_path,'output.txt')
    f = open(out_txtsavedpath, 'w+')
    
    print('Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Data path: {}, Saved path: {}'.format(
        args.seed, args.epoch, args.b, args.lr, args.weight_d, args.gpu, args.data_path, args.save_path),
        file=f)
    
    print('index: {}; maximum value of validation accuracy: {}.'.format(Valid_Accuracy.index(max(Valid_Accuracy))+1, max(Valid_Accuracy)), file=f)
    print('index: {}; maximum value of validation f1-score: {}.'.format(f1_s.index(max(f1_s))+1, max(f1_s)), file=f)
    print('--------------------------------------------------')
    print('Validation accuracy: {}'.format(Valid_Accuracy), file=f)
    print('Validation F1-score: {}'.format(f1_s), file=f)
    
    ######load the best trained model and test testing data  ，测试函数，推理
    best_net = get_network(args)
    best_net.load_state_dict(torch.load(best_weights_path))
    best_net.load_state_dict(torch.load(best_weights_path))
    
    total_num_paras, trainable_num_paras = get_parameter_number(best_net)
    print('The total number of network parameters = {}'.format(total_num_paras), file=f)
    print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)
    
    best_net.eval()
    number = 0
    correct_test = 0.0
    test_target =[]
    test_predict = []
    
    with torch.no_grad():
        
        start = time.time()
        
        for n_iter, (image, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                labels = labels.cuda()

            output = best_net(image)
            output = torch.softmax(output, dim= 1)
            preds = torch.argmax(output, dim =1)
            # _, preds = output.topk(5, 1, largest=True, sorted=True)
            # _, preds = output.max(1)
            correct_test += preds.eq(labels).sum()
            
            if args.gpu:
                labels = labels.cpu()
                preds = preds.cpu()
        
            test_target.extend(labels.numpy().tolist())
            test_predict.extend(preds.numpy().tolist())
        
            number +=1
        
        print('Label values: {}'.format(test_target), file=f)
        print('Predicted values: {}'.format(test_predict), file=f)

        finish = time.time()
        accuracy_test = correct_test.float() / len(test_loader.dataset)
        print('Testing network......', file=f)
        print('Test set: Accuracy: {:.5f}, Time consumed: {:.5f}s'.format(
            accuracy_test,
            finish - start
            ), file=f)
        
        #Obtain f1_score of the prediction
        fs_test = f1_score(test_target, test_predict, average='macro')
        print('f1 score = {:.5f}'.format(fs_test), file=f)
        
        kappa_value = cohen_kappa_score(test_target, test_predict)
        print("kappa value = {:.5f}".format(kappa_value), file=f)
        
        precision_test = precision_score(test_target, test_predict, average='macro', zero_division=0)
        print('precision = {:.5f}'.format(precision_test), file=f)
        
        recall_test = recall_score(test_target, test_predict, average='macro', zero_division=0)
        print('recall = {:.5f}'.format(recall_test), file=f)
        
        #Output the classification report
        print('------------', file=f)
        print('Classification Report', file=f)
        print(classification_report(test_target, test_predict, zero_division=0), file=f)
        
        if not os.path.exists('./results.csv'):
            with open("./results.csv", 'w+') as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(['index','accuracy','f1-score','precision','recall','kappa','time_consumed'])
        
        with open("./results.csv", 'a+') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([args.seed, accuracy_test, fs_test, precision_test, recall_test, kappa_value, finish-start])
        
        Class_labels = ['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
        #Show the confusion matrix so that it can help us observe the results more intuitively
        def show_confusion_matrix(validations, predictions):
            matrix = confusion_matrix(validations, predictions) #No one-hot
            #matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix,
                  cmap="coolwarm",
                  linecolor='white',
                  linewidths=1,
                  xticklabels=Class_labels,
                  yticklabels=Class_labels,
                  annot=True,
                  fmt="d")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_figuresavedpath = os.path.join(checkpoint_path,'Confusion_matrix.png')
            plt.savefig(cm_figuresavedpath)

        show_confusion_matrix(test_target, test_predict)
    
    if args.gpu:
        print('GPU INFO.....', file=f)
        print(torch.cuda.memory_summary(), end='', file=f)

    # 可选：添加dropout
    net.apply(lambda m: setattr(m, 'dropout', nn.Dropout(p=0.2)) if hasattr(m, 'dropout') else None)