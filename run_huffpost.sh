#!/usr/bin/bash
#SBATCH --job-name=huffpost
#SBATCH --output=logs/huffpost.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=tir-0-32,tir-0-36
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00


DATASET="huffpost";

OFFLINE_STEPS=6000;
WARMSTART_STEPS=3000;
OFFLINE_SWA_STEPS=500;
ONLINE_STEPS=500;
SEED=2;

source activate wild-time;

########## Baselines ##########

# ERM Reproduction
python main.py \
    --train --eval_fix --exp_name erm_baseline_seed_${SEED} \
    --dataset $DATASET --method erm --random_seed $SEED \
    --offline_steps $OFFLINE_STEPS;

# ERM + Finetuning
python main.py \
    --train --eval_warmstart_finetune --exp_name erm_ft_seed_${SEED} \
    --dataset $DATASET --method erm --random_seed $SEED \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Finetuning
python main.py \
    --train --eval_warmstart_finetune --exp_name ft_seed_${SEED} \
    --dataset $DATASET --method erm --random_seed $SEED \
    --offline_steps 0 --online_steps $OFFLINE_STEPS;

# EWC
python main.py \
    --train --eval_fix --exp_name ewc_seed_${SEED} \
    --dataset $DATASET --method ewc --random_seed $SEED \
    --offline_steps $OFFLINE_STEPS --online_steps $ONLINE_STEPS;


# SWA
python main.py \
    --train --eval_fix --exp_name swa_seed_${SEED} \
    --dataset $DATASET --method swa --swa_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
    --offline_steps $OFFLINE_STEPS --online_steps $ONLINE_STEPS;

# ERM + SWA FT
python main.py \
    --train --eval_warmstart_finetune --exp_name erm_ordered_swa_seed_${SEED} \
    --dataset $DATASET --method swa --swa_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# EWA
python main.py \
    --train --eval_fix --exp_name unordered_ewa_seed_${SEED} \
    --dataset $DATASET --method swa --swa_ewa --swa_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
    --offline_steps $OFFLINE_STEPS --online_steps 0;


# EWA (Shuffle Buckets)
python main.py \
    --train --eval_warmstart_finetune --shuffle_timesteps  --exp_name erm_shuffled_ewa_seed_${SEED} \
    --dataset $DATASET --method swa --swa_ewa --random_seed $SEED \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;


########## Proposal ##########
# Warm Start and EWA
python main.py \
    --train --eval_warmstart_finetune --exp_name warmstart_csaw_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.50 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Cold Start and EWA
python main.py \
    --train --eval_warmstart_finetune --exp_name coldstart_csaw_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.50 \
    --offline_steps 0 --online_steps $ONLINE_STEPS;

######### EWA Decay ##########

# Lamba 0.1
python main.py \
    --train --eval_warmstart_finetune  --exp_name csaw_lambda0.1_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.10 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Lamba 0.25
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_lambda0.25_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.25 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Lamba 0.50: Proposed Method

# Lambda 0.75
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_lambda0.75_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.75 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Lamba 0.90
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_lambda0.90_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.90 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;


######### Comparisons ##########

# SAM Optimizer
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_sam_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.50 --sam \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

#########  K-Last Weight Averaging #########
# Last 10% of Timestamps
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_last0.1_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.50 --last_k_timesteps 0.1 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Last 25% of Timestamps
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_last0.25_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.50 --last_k_timesteps 0.25 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# Last 50% of Timestamps
python main.py \
    --train --eval_warmstart_finetune --exp_name csaw_last0.50_seed_${SEED} \
    --dataset $DATASET --random_seed $SEED \
    --method swa --swa_ewa --swa_ewa_lambda 0.50 --last_k_timesteps 0.50 \
    --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;