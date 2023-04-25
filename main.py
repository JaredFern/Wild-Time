import argparse
import os
import random

import numpy as np
import torch
from torch import cuda

from wildtime.baseline_trainer import logger_init, trainer_init
from wildtime.configs.eval_fix import configs_yearbook, configs_huffpost, configs_fmow, configs_arxiv, configs_mimic_mortality, configs_mimic_readmission
from wildtime.methods.agem.agem import AGEM
from wildtime.methods.coral.coral import DeepCORAL
from wildtime.methods.erm.erm import ERM
from wildtime.methods.ewc.ewc import EWC
from wildtime.methods.ft.ft import FT
from wildtime.methods.groupdro.groupdro import GroupDRO
from wildtime.methods.irm.irm import IRM
from wildtime.methods.si.si import SI
from wildtime.methods.simclr.simclr import SimCLR
from wildtime.methods.swa.swa import SWA
from wildtime.methods.swav.swav import SwaV

device = 'cuda' if cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='arxiv', choices=['arxiv', 'huffpost', 'fmow', 'yearbook', 'mimic_mortality', 'mimic_readmission'])
    parser.add_argument('--method', type=str, default='erm', choices=['erm', 'ft', 'ewc', 'si', 'irm', 'coral', 'groupdro', 'agem', 'simclr', 'swav', 'swa'])
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()


    experimental_params = {
        'device': 0,
        'random_seed': 1,
        'num_workers': 8,
        # 'mini_batch_size': 128,
        'eval_batch_size': 512,
        # 'ssl_finetune_iter': 1500,
        'linear_probe_iter': None,
        'train_update_iter': 250,
        'offline_steps': 4000,
        'online_steps': 250,

        'eval_fix': False,
        'eval_warmstart_finetune': True,
        'eval_fixed_timesteps': [],
        'eval_features': False,
        'difficulty': False,
        'linear_probe': False,
        # 'online': False,

        'load_model': False,
        'torch_compile': False,
        'time_conditioned': False,
        'swa_ewa': True,
        'swa_ewa_lambda': 0.1,
        'swa_steps': None,
        'ewc_task_decay': 2,

        # Feature Analysis Params
        'feat_threshold': 0.1,
        'feat_num_samples': 100,
        'feat_num_components': 50,
        'feat_split': 'test',

        'checkpoint_path': None,
        'data_dir': '/data/datasets/wilds/',
        'log_dir': '/data/jaredfer/wilds-time/results',
        'results_dir': '/data/jaredfer/wilds-time/results',
    }
    configs = {
        **getattr(
            globals()[f"configs_{args.dataset}"],
            f"configs_{args.dataset}_{args.method}"
        ),
        **experimental_params
    }
    configs = argparse.Namespace(**configs)

    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.cuda.manual_seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    if not os.path.isdir(configs.data_dir):
        raise ValueError(f'Data directory {configs.data_dir} does not exist!')
    if configs.load_model and not os.path.isdir(configs.log_dir):
        raise ValueError(f'Model checkpoint directory {configs.log_dir} does not exist!')
    if not os.path.isdir(configs.results_dir):
        raise ValueError(f'Results directory {configs.results_dir} does not exist!')

    if configs.method in ['groupdro', 'irm']:
        configs.reduction = 'none'

    dataset, criterion, network, optimizer, scheduler = trainer_init(configs)

    if configs.method == 'groupdro':
        trainer = GroupDRO(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'coral':
        trainer = DeepCORAL(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'irm':
        trainer = IRM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'ft':
        trainer = FT(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'erm':
        trainer = ERM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'ewc':
        trainer = EWC(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'agem':
        trainer = AGEM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'si':
        trainer = SI(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'simclr':
        trainer = SimCLR(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'swav':
        trainer = SwaV(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'swa':
        trainer = SWA(configs, dataset, network, criterion, optimizer, scheduler)
    else:
        raise ValueError

    logger_init(configs)
    trainer.run()

    # todo: When using a dictionary to store classes, each class will be instantiated and there will be incompatible datasets and methods
    # if configs.method in ['coral', 'groupdro', 'irm']:
    #     trainer_dict = {
    #         'groupdro': GroupDRO(*param),
    #         'coral':    DeepCORAL(*param),
    #         'irm':      IRM(*param),
    #     }
    #
    # else:
    #     trainer_dict = {
    #                         'ft':     FT(*param),
    #                         'erm':    ERM(*param),
    #                         'ewc':    EWC(*param),
    #                         'agem':   AGEM(*param),
    #                         'si':     SI(*param),
    #                         'simclr': SimCLR(*param),
    #                         'swav':   SwaV(*param),
    #                         'swa':    SWA(*param),
    #     }




