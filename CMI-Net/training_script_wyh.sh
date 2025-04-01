#!/bin/bash
#SBATCH -J training
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_1d2g
#SBATCH -c 2
#SBATCH -N 1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

#source /home/senmaoye/.bashrc

#cd jay/multitask

#conda activate torch

nvidia-smi

# 定义时间变量
current_time=$(date "+%Y-%m-%d_%H_%M_%S")
echo "当前时间是: $current_time"

# 定义基础保存路径
base_path="results/${current_time}"
mkdir -p $base_path

# 运行每个fold的训练
python train5_ConvTrans_auto.py  --data_path './data/000goat.pt' --save_path "${base_path}/setting000"
python train5_ConvTrans_auto.py  --data_path './data/001goat.pt' --save_path "${base_path}/setting001"
python train5_ConvTrans_auto.py  --data_path './data/002goat.pt' --save_path "${base_path}/setting002"
python train5_ConvTrans_auto.py  --data_path './data/003goat.pt' --save_path "${base_path}/setting003"
python train5_ConvTrans_auto.py  --data_path './data/004goat.pt' --save_path "${base_path}/setting004"
python train5_ConvTrans_auto.py  --data_path './data/005goat.pt' --save_path "${base_path}/setting005"

# 运行汇总脚本，保存在同一个时间文件夹下
python 五折汇总.py --save_path "${base_path}/summary"
