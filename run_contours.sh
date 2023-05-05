#!/usr/bin/bash
LOG_DIR=" /projects/tir6/strubell/jaredfer/projects/wild-time/results"
EXP_NAME="0426_fmow_swa_eval_warmstart_finetune_3_1"
METHOD="swa" # or erm
DATASET="fmow"

# Outputs png and pdf to ${LOG_DIR}/${EXP_NAME}/loss_contours
python main.py --loss_contours \
	--dataset DATASET --method METHOD # "erm" or "swa" \
	--exp_name EXP_NAME \
  --contour_timesteps T0 --contour_timesteps T1 # Repeat for timesteps you want to plot \ 
  --contour_models "T0" "T0_swa" "T1" \
	--contour_margin 0.1 --contour_granularity 5 --contour_increment 0.01;
