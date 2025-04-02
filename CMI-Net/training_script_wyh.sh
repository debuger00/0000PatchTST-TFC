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

current_time=$(date "+%Y-%m-%d_%H_%M_%S")
echo "当前时间是: $current_time"


python train5_ConvTrans_auto.py  --data_path './data/000goat.pt' --save_path "setting000/${current_time}"
python train5_ConvTrans_auto.py  --data_path './data/001goat.pt' --save_path "setting001/${current_time}"
python train5_ConvTrans_auto.py  --data_path './data/002goat.pt' --save_path "setting002/${current_time}"
python train5_ConvTrans_auto.py  --data_path './data/003goat.pt' --save_path "setting003/${current_time}"
python train5_ConvTrans_auto.py  --data_path './data/004goat.pt' --save_path "setting004/${current_time}"
python train5_ConvTrans_auto.py  --data_path './data/005goat.pt' --save_path "setting005/${current_time}"

python 五折汇总.py --save_path "setting999${current_time}" --time ${current_time}
