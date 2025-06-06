"""Updated implementation for TF-C -- Xiang Zhang, Jan 16, 2023"""

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
from model import *
from dataloader import data_generator
from trainer import Trainer




# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd() ##加入 main.py 的目录, 因为只有在 main.py 的目录下才能找到 config_files 目录，才能运行
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description') 
parser.add_argument('--seed', default=42, type=int, help='seed value')

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='pre_train', type=str,
                    help='pre_train, fine_tune_test')

parser.add_argument('--pretrain_dataset', default='AAR', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG,AAR1') ##预训练数据集选择
parser.add_argument('--target_dataset', default='AAR', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG,AAR2') ##微调数据集选择

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory') ##日志保存路径
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda') ##设备选择
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory') ##项目主目录
args, unknown = parser.parse_known_args()

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(pretrain_dataset) + '_2_' + str(targetdata) ##实验描述字符串


method = 'TF-C' ##方法选择
training_mode = args.training_mode ##训练模式选择
run_description = args.run_description ##运行描述字符串
logs_save_dir = args.logs_save_dir ##日志保存路径
os.makedirs(logs_save_dir, exist_ok=True) ##创建日志保存路径
exec(f'from config_files.{pretrain_dataset}_Configs import Config as Configs') ##加载预训练数据集配置
configs = Configs() ##配置选择

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################
 
experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_2layertransformer")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0  # 计数器,用于记录当前的训练轮数 没有使用
src_counter = 0  # 源域计数器,用于记录源域的样本数量 没有使用


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets#################################################### 加载数据集
sourcedata_path = f"../../datasets/{pretrain_dataset}"
targetdata_path = f"../../datasets/{targetdata}"
subset = True  # if subset= true, use a subset for debugging.
# train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = subset)
train_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = subset)
test_dl = None
valid_dl = None

logger.debug("Data loaded ...")

# Load Model#################################################### 加载模型
"""Here are two models, one basemodel, another is temporal contrastive model"""
TFC_model = TFC(configs).to(device)
classifier = target_classifier(configs).to(device)
temporal_contr_model = None


if training_mode == "fine_tune_test":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{SEED}_2layertransformer", "saved_models"))  ##加载预训练模型
    print("The loading file path", load_from)
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)

model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainer
Trainer(TFC_model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode)

logger.debug(f"Training time is : {datetime.now()-start_time}")
