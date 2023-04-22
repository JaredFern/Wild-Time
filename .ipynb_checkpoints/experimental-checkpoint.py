from wildtime import baseline_trainer, dataloader
from wildtime.configs.eval_fix import configs_yearbook, configs_huffpost, configs_fmow, configs_arxiv, configs_mimic_mortality, configs_mimic_readmission

import argparse

experimental_params = {
    'device': 0,
    'random_seed': 1,
    'num_workers': 16,
    'mini_batch_size': 128,
    'eval_batch_size': 512,
    'ssl_finetune_iter': 1500,
    'linear_probe_iter': 0,
    'train_update_iter': 3000,
    'offline_steps': 3000,
    # 'online_steps': 500,

    'eval_fix': True,
    'eval_warmstart_finetune': False,
    # 'eval_fixed_timesteps': [9,10],
    'eval_features': False,
    'difficulty': False,
    'linear_probe': False,
    'online': False,
    # 'split_time': 10,
    'eval_next_timestamps': 6,
    'eval_all_timestamps': False,

    'load_model': False,
    'torch_compile': False,
    'time_conditioned': False,
    'swa_ewa': True,
    'swa_ewa_lambda': 0.5,
    'swa_steps': None,

    # Feature Analysis Params
    'feat_threshold': 0.1,
    'feat_num_samples': 100,
    'feat_num_components': 50,
    'feat_split': 'test',

    'data_dir': '/compute/tir-1-11/jaredfer/wilds/data/mimiciv',
    'log_dir': './checkpoints',
    'checkpoint_path': None,
    'results_dir': './results'
}

config = {**configs_mimic_mortality.configs_mimic_erm, **experimental_params}

experimental_config = argparse.Namespace(**config)

baseline_trainer.train(experimental_config)
# if configs.train_model:
#     baseline_trainer.train(experimental_config)
# elif configs.loss_analysis:
#     pass
# elif configs.feature_analysis:
#     pass
