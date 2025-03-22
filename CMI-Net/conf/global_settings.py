""" configurations for this project

author axiumao
"""

from datetime import datetime

#directory to save weights file
# 保存模型权重文件的目录
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [0, 60, 80] # 学习率调整的里程碑

#initial learning rate
#INIT_LR = 0.1

# DATE_FORMAT = '%d_%B_%Y_%Hh_%Mm_%Ss'
DATE_FORMAT = '%Y_%m_%d_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir 保存训练日志的目录
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








