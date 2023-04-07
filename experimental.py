from wildtime import baseline_trainer, dataloader
from wildtime.configs.eval_fix import configs_yearbook, configs_arxiv, configs_fmow

import argparse

experimental_params = {
    'device': 0,
    'random_seed': 1,
    'num_workers': 32,
    'mini_batch_size': 64,
    'eval_batch_size': 512,
    'ssl_finetune_iter': 1500,
    'linear_probe_iter': 0,
    'train_update_iter': 500,
    'offline_steps': 1000,
    'online_steps': 250,

    'eval_fix': True,
    'difficulty': False,
    'linear_probe': False,
    'online': False,
    'split_time': 13,
    'eval_next_timestamps': 6,
    'eval_all_timestamps': False,

    'load_model': False,
    'torch_compile': False,
    'time_conditioned': False,
    'swa_ewa': True,
    'swa_ewa_lambda': 0.5,
    'swa_steps': None,

    'data_dir': '/compute/tir-1-11/jaredfer/wilds/data',
    'log_dir': './checkpoints',
    # 'checkpoint_path': 'checkpoints/fmow_SWA-train_update_iter=500-lr=0.0001-mini_batch_size=64-seed=1-eval_fix_time=13_swa',
    'checkpoint_path': None,
    'results_dir': './results'
}

config = {**configs_fmow.configs_fmow_swa, **experimental_params}

experimental_config = argparse.Namespace(**config)

baseline_trainer.train(experimental_config)
