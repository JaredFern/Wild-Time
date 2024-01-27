#!/usr/bin/bash
#SBATCH --job-name=forgetting
#SBATCH --output=logs/forgetting-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00

source activate wild-time

CUDA_VISIBLE_DEVICES=0
RESULTS_DIR="/data/tir/projects/tir6/strubell/jaredfer/projects/wild-time/results/icml"

DATASET=$1
EXP_NAME=$2
SEED=0; $SLURM_ARRAY_TASK_ID


if [ $DATASET = 'arxiv' ]; then
    BATCH_SIZE=2048
elif [ $DATASET = "huffpost" ]; then
    BATCH_SIZE=2048
elif [ $DATASET = "fmow" ]; then
    BATCH_SIZE=2048
elif [ $DATASET = "yearbook" ]; then
    BATCH_SIZE=256
fi;
# elif [ $1 = "mimic" ]; then
# fi;

for LAMBDA in 0 1.0
do
    python3 forgetting_main.py \
        --dataset $DATASET \
        --results_dir $RESULTS_DIR \
        --exp_path $EXP_NAME \
        --method swa \
        --eval_batch_size $BATCH_SIZE \
        --swa_ewa_lambda $LAMBDA \
        --random_seed $SEED
done
