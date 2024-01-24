#!/usr/bin/bash
#SBATCH --job-name=forgetting
#SBATCH --output=logs/forgetting.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00

# CHECKPOINT_DIR='/data/tir/projects/tir6/strubell/jaredfer/projects/wild-time/results/0504_fmow_swa_coldstart_csaw/checkpoints'

CUDA_VISIBLE_DEVICES=0
DATA_DIR="/data/tir/projects/tir6/strubell/jaredfer/projects/wild-time/results"

# Experimental Dir Paths
DATASET="arxiv"
BATCH_SIZE=2048

# DECAY_FACTOR=0.50
# EXP_DIR="0509_arxiv_swa_warmstart_csaw_seed_1"

DECAY_FACTOR=0.75
EXP_DIR="0509_arxiv_swa_coldstart_csaw_seed_1"


# DATASET="huffpost"
# BATCH_SIZE=256
# DECAY_FACTOR=0.85
# EXP_DIR="0504_huffpost_swa_coldstart_csaw"

# DECAY_FACTOR=0.45
# EXP_DIR='0504_huffpost_swa_warmstart_csaw_2'


# DATASET="fmow"
# BATCH_SIZE=256

# DECAY_FACTOR=0.75
# EXP_DIR='0504_fmow_swa_csaw_lambda0.75'

# DECAY_FACTOR=0.90
# EXP_DIR="0504_fmow_swa_coldstart_csaw"

# DATASET="yearbook"
# BATCH_SIZE=256

# # DECAY_FACTOR=0.90
# # EXP_DIR="0504_yearbook_swa_coldstart_csaw"

# DECAY_FACTOR=0.75
# EXP_DIR="0504_yearbook_swa_csaw_lambda0.75"

source activate wild-time
for LAMBDA in $DECAY_FACTOR 1.0
do
    python3 forgetting_main.py \
        --dataset $DATASET \
        --method swa \
        --model $DATASET-$METHOD \
        --exp_path ${EXP_DIR} \
        --eval_batch_size $BATCH_SIZE \
        --swa_ewa_lambda $LAMBDA 
done
