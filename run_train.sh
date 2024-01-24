#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --out=logs/%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00

if [ $1 = "fmow" ]; then
    DATASET="fmow";
    OFFLINE_STEPS=3000;
    OFFLINE_SWA_STEPS=500;
    WARMSTART_STEPS=3000;
    ONLINE_STEPS=500;
    WARMSTART_CKPT_PATH="";
elif [ $1 = "huffpost" ]; then
    DATASET="huffpost";
    OFFLINE_STEPS=3000;
    WARMSTART_STEPS=3000;
    OFFLINE_SWA_STEPS=500;
    ONLINE_STEPS=500;
    WARMSTART_CKPT_PATH="";
elif [ $1 = "arxiv" ]; then
    DATASET="arxiv";
    OFFLINE_STEPS=6000;
    WARMSTART_STEPS=6000;
    OFFLINE_SWA_STEPS=1000;
    ONLINE_STEPS=1000;
    WARMSTART_CKPT_PATH="";
elif [ $1 = "yearbook" ]; then
    DATASET="yearbook";
    OFFLINE_STEPS=3000;
    WARMSTART_STEPS=3000;
    OFFLINE_SWA_STEPS=500;
    ONLINE_STEPS=500;
    WARMSTART_CKPT_PATH="";
elif [ $1 = "mimic_m" ]; then
    DATASET="mimic_mortality";
    OFFLINE_STEPS=3000;
    WARMSTART_STEPS=3000;
    OFFLINE_SWA_STEPS=500;
    ONLINE_STEPS=500;
    WARMSTART_CKPT_PATH="";
fi;


METHOD=$2
SEED=$SLURM_ARRAY_TASK_ID


source activate wild-time;

python main.py \
    --train --eval_fix --method $METHOD  --exp_name ${METHOD}_${SEED} --dataset $DATASET \
    --offline_steps $OFFLINE_STEPS --online_steps $ONLINE_STEPS  \
    --random_seed $SEED;

# python main.py \
#     --train --eval_fix --shuffle_timesteps  --exp_name swa_baseline_${SEED} \
#     --dataset $DATASET --method swa --swa_offline_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
#     --offline_steps $OFFLINE_STEPS;
# # EWA (Shuffle Buckets)
# python main.py \
#     --train --eval_warmstart_finetune --shuffle_timesteps  --exp_name erm_shuffled_ewa_seed_${SEED} \
#     --dataset $DATASET --method swa --swa_ewa --random_seed $SEED \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

#     python main.py \
#         --train --eval_fix --exp_name erm_baseline_seed_${SEED} \
#         --dataset $DATASET --method erm --random_seed $SEED \
#         --offline_steps $OFFLINE_STEPS;

#     # Ordered SWA
#     python main.py \
#         --train --eval_warmstart_finetune --exp_name ordered_swa_seed_${SEED} \
#         --dataset $DATASET --method swa --random_seed $SEED \
#         --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

#     # EWA (Shuffle Buckets)
#     python main.py \
#         --train --eval_warmstart_finetune --shuffle_timesteps  --exp_name shuffled_ewa_seed_${SEED} \
#         --dataset $DATASET --method swa --swa_ewa --random_seed $SEED \
#         --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

#     python main.py \
#         --train --eval_fix --exp_name irm_seed_${SEED} \
#         --dataset $DATASET --method irm --random_seed $SEED \
#         --offline_steps $OFFLINE_STEPS;

    # python main.py \
    #     --train --eval_fix --exp_name dro_seed_${SEED} \
    #     --dataset $DATASET --method groupdro --random_seed $SEED \
    #     --offline_steps $OFFLINE_STEPS;

#     python main.py \
#         --train --eval_warmstart_finetune --exp_name warmstart_csaw_seed_${SEED}_lambda_0.10 \
#         --dataset $DATASET --random_seed $SEED \
#         --method swa --swa_ewa --swa_ewa_lambda 0.10 \
#         --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;


# # ERM + Finetuning
# python main.py \
#     --train --eval_warmstart_finetune --exp_name erm_ft_seed_${SEED} \
#     --dataset $DATASET --method erm --random_seed $SEED \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Finetuning
# python main.py \
#     --train --eval_warmstart_finetune --exp_name ft_fixed_seed_${SEED} \
#     --dataset $DATASET --method erm --random_seed $SEED \
#     --offline_steps 0 --online_steps $ONLINE_STEPS;

# # EWA (Shuffle Buckets)
# python main.py \
#     --train --eval_warmstart_finetune --shuffle_timesteps  --exp_name erm_shuffled_ewa_seed_${SEED} \
#     --dataset $DATASET --method swa --swa_ewa --random_seed $SEED \
#      --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# done;
# # EWC
# python main.py \
#     --train --eval_fix --exp_name ewc_seed_${SEED} \
#     --dataset $DATASET --method ewc --random_seed $SEED \
#     --offline_steps $OFFLINE_STEPS --online_steps $ONLINE_STEPS;


# # SWA
# python main.py \
#     --train --eval_fix --exp_name swa_seed_${SEED} \
#     --dataset $DATASET --method swa --swa_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
#     --offline_steps $OFFLINE_STEPS --online_steps $ONLINE_STEPS;

# # ERM + SWA FT
# python main.py \
#     --train --eval_warmstart_finetune --exp_name erm_ordered_swa_seed_${SEED} \
#     --dataset $DATASET --method swa --swa_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # EWA
# python main.py \
#     --train --eval_fix --exp_name unordered_ewa_seed_${SEED} \
#     --dataset $DATASET --method swa --swa_ewa --swa_steps $OFFLINE_SWA_STEPS --random_seed $SEED \
#     --offline_steps $OFFLINE_STEPS --online_steps 0;





# ########## Proposal ##########
# # Warm Start and EWA
# python main.py \
#     --train --eval_warmstart_finetune --exp_name warmstart_csaw_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.50 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Cold Start and EWA
# python main.py \
#     --train --eval_warmstart_finetune --exp_name coldstart_csaw_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.50 \
#     --offline_steps 0 --online_steps $ONLINE_STEPS;

# ######### EWA Decay ##########

# # Lamba 0.1
# python main.py \
#     --train --eval_warmstart_finetune  --exp_name csaw_lambda0.1_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.10 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Lamba 0.25
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_lambda0.25_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.25 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Lamba 0.50: Proposed Method

# # Lambda 0.75
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_lambda0.75_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.75 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Lamba 0.90
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_lambda0.90_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.90 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;


# ######### Comparisons ##########

# # SAM Optimizer
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_sam_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.50 --sam \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# #########  K-Last Weight Averaging #########
# # Last 10% of Timestamps
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_last0.1_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.50 --last_k_timesteps 0.1 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Last 25% of Timestamps
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_last0.25_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.50 --last_k_timesteps 0.25 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;

# # Last 50% of Timestamps
# python main.py \
#     --train --eval_warmstart_finetune --exp_name csaw_last0.50_seed_${SEED} \
#     --dataset $DATASET --random_seed $SEED \
#     --method swa --swa_ewa --swa_ewa_lambda 0.50 --last_k_timesteps 0.50 \
#     --offline_steps $WARMSTART_STEPS --online_steps $ONLINE_STEPS;