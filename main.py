import argparse
import json
import logging
import os
import random
import sys
from datetime import date

import numpy as np
import torch
from torch import cuda

from wildtime.methods import loss_landscape, interpolation_plots
from wildtime.baseline_trainer import trainer_init
from wildtime.configs.eval_fix import configs_yearbook, configs_huffpost, configs_fmow, configs_arxiv, configs_mimic_mortality, configs_mimic_readmission
from wildtime.optimizers.sam import SAM
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

def logger_init(args, train=False):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Experiment Name: Date_Dataset_Method_Setting/
    if args.eval_fix:
        eval_setting = "eval_fix"
    elif args.eval_warmstart_finetune:
        eval_setting = "eval_warmstart_finetune"
    elif args.eval_fixed_timesteps:
        eval_setting = "eval_fixed_timesteps"
    else:
        eval_setting = 'eval_stream'

    if not hasattr(args, 'exp_name') or args.exp_name is None:
        args.exp_name = f"{date.today().strftime('%m%d')}_{args.dataset}_{args.method}_{eval_setting}"
    else:
        args.exp_name = f"{date.today().strftime('%m%d')}_{args.dataset}_{args.method}_{args.exp_name}"

    if not hasattr(args, 'exp_path') or args.exp_path is None:
        args.exp_path = os.path.join(args.log_dir, args.exp_name)
    else:
        args.exp_path = os.path.join(args.log_dir, args.exp_path)

    args.model_path = f"{args.exp_path}/checkpoints"

    if train:
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        else:
            i = 1
            while True:
                new_exp_path = f"{args.exp_path}_{i}"
                if not os.path.exists(new_exp_path):
                    os.makedirs(new_exp_path)
                    args.exp_path = new_exp_path
                    break
                i += 1

        args.model_path = f"{args.exp_path}/checkpoints"
        os.makedirs(args.model_path)

    # Create stdout logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{args.exp_path}/log.out', 'a')],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    # Save Config as json file
    json_str = json.dumps(vars(args))
    with open(f"{args.exp_path}/config.json", "w") as file_handler:
        file_handler.write(json_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Functionality
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--loss_contours', action='store_true')
    parser.add_argument('--interpolation_plots', action='store_true')

    # Dataset and Methods
    parser.add_argument('--dataset', type=str, default='arxiv', choices=['arxiv', 'huffpost', 'fmow', 'yearbook', 'mimic_mortality', 'mimic_readmission'])
    parser.add_argument('--method', type=str, default='erm', choices=['erm', 'ft', 'ewc', 'si', 'irm', 'coral', 'groupdro', 'agem', 'simclr', 'swav', 'swa'])

    # Configs
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--exp_path', type=str, default=None)

    # Experimental Setting
    parser.add_argument('--offline_steps', type=int, default=3000)
    parser.add_argument('--online_steps', type=int, default=250)
    parser.add_argument('--eval_fix', action='store_true')
    parser.add_argument('--eval_stream', action='store_true')
    parser.add_argument('--eval_warmstart_finetune', action='store_true')
    parser.add_argument('--eval_fixed_timesteps', action='append', default=[])
    parser.add_argument('--shuffle_timesteps', action='store_true')
    parser.add_argument('--online_timesteps', action='append', default=[])
    parser.add_argument('--last_k_timesteps', type=float, default=1)
    parser.add_argument('--random_seed', type=int, default=0)

    # Experimental Parameters
    parser.add_argument('--swa_ewa', action='store_true')
    parser.add_argument('--swa_steps', default=None)
    parser.add_argument('--swa_ewa_lambda', type=float, default=0.5)
    parser.add_argument('--ewc_task_decay', type=float, default=1.0)
    parser.add_argument('--sam', action='store_true')

    # Contour Parameters
    parser.add_argument('--contour_timesteps', action='append', type=int)
    parser.add_argument('--contour_models', nargs=3)
    parser.add_argument('--contour_granularity', type=int, default=5)
    parser.add_argument('--contour_margin', type=float, default=0.1)
    parser.add_argument('--contour_increment', type=float, default=0.01)

    # Interpolation Parameters
    parser.add_argument('--interpolation_timesteps', nargs='+', type=int)
    parser.add_argument('--interpolation_models', nargs=2)
    parser.add_argument('--interpolation_granularity', type=int, default=5)
    args = parser.parse_args()

    experimental_params = {
        'device': 0,
        'random_seed': 0,
        'num_workers': 8,
        # 'mini_batch_size': 128,
        'eval_batch_size': 512,
        'linear_probe_iter': None,
        'linear_probe': False,

        'load_model': False,
        'torch_compile': False,
        'time_conditioned': False,

        # Feature Analysis Params
        'feat_threshold': 0.1,
        'feat_num_samples': 100,
        'feat_num_components': 50,
        'feat_split': 'test',

        # Loss Contour Params
        'contour_timesteps': [11],
        'contour_models': ['9', '10', '10_swa'],
        'contour_granularity': 5,
        'contour_metric': 'losses',
        'contour_increment': 0.01,
        'contour_margin': 0.05,

        'interpolation_timesteps': [11],
        'interpolation_models': ['9', '10', '10_swa'],
        'interpolation_granularity': 5,
        'interpolation_metric': 'losses',

        'checkpoint_path': None,
        'data_dir': '/projects/tir6/strubell/data/wilds/data',
        'log_dir': '/projects/tir6/strubell/jaredfer/projects/wild-time/results',
        'results_dir': '/projects/tir6/strubell/jaredfer/projects/wild-time/results',
    }
    configs = {
        **getattr(
            globals()[f"configs_{args.dataset}"],
            f"configs_{args.dataset}_{args.method}"
        ),
        **experimental_params,
        **vars(args)
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
    if configs.torch_compile:
        network = torch.compile(network)
    if configs.sam:
        optimizer = SAM(network.parameters(), torch.optim.Adam, rho=0.05, adaptive=False)

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

    logger_init(configs, configs.train)
    logging.info(configs)

    if configs.train:
        trainer.run()
    if configs.eval:
        trainer.eval()
    if configs.loss_contours:
        contour_data = loss_landscape.generate_loss_contours(configs, trainer)
        contour_dir = os.path.join(configs.exp_path, 'loss_contours')
        if 'model_ids' in contour_data.keys():
            labels = contour_data['model_ids']
        else:
            labels = ['w_0', 'w_1', 'w_2']
    if configs.interpolation_plots:
        interpolation_data = interpolation_plots.generate_interpolation_plots(configs, trainer)
        print(interpolation_data)

        loss_landscape.plot_contour(
            contour_data['grid'],
            contour_data[configs.contour_metric],
            contour_data['coords'],
            labels,
            contour_dir,
            contour_data['title'],
            configs.contour_increment,
            configs.contour_margin
        )

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




