import copy
import logging
import math
import os

import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from ..dataloaders import FastDataLoader,InfiniteDataLoader
from ..base_trainer import BaseTrainer
# from torchcontrib.optim import SWA as SWA_optimizer
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Subset

logger = logging.getLogger(__name__)

class SWA(BaseTrainer):
    """
    Stochastic Weighted Averaging

    Original paper:
        @article{izmailov2018averaging,
            title={Averaging weights leads to wider optima and better generalization},
            author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
            journal={arXiv preprint arXiv:1803.05407},
            year={2018}
        }
    """
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler, update_bn_on_save=False):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.network = self.network
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=args.lr, anneal_epochs=1)
        self.swa_ewa_lambda = args.swa_ewa_lambda
        self.update_bn_on_save = update_bn_on_save

        if args.swa_ewa:
            self.ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
                (1 - args.swa_ewa_lambda) * averaged_model_parameter + args.swa_ewa_lambda * model_parameter
            self.swa_model = AveragedModel(self.network, avg_fn=self.ema_avg)
        else:
            self.swa_model = AveragedModel(self.network)

        if args.swa_load_from_checkpoint:
            # Initialize weights with the offline ckpt: Will either be random, pretrained, warmstart init
            model_path = os.path.join(self.args.results_dir, self.args.exp_path, 'checkpoints')
            if os.path.exists(os.path.join(model_path, 'time_offline_swa.pth')):
                init_weights = torch.load(os.path.join(model_path, f'time_offline_swa.pth'))
                self.swa_model.load_state_dict(init_weights, strict=False)
                self.swa_model.avg_fn = self.ema_avg
            elif os.path.exists(os.path.join(model_path, 'time_offline.pth')):
                init_weights = torch.load(os.path.join(model_path, f'time_offline.pth'))
                self.network.load_state_dict(init_weights, strict=False)
                self.swa_model = AveragedModel(self.network, avg_fn=self.ema_avg)

            for timestep in self.train_dataset.ENV:
                ckpt_path = os.path.join(model_path, f'time_{timestep}.pth')
                weights = torch.load(ckpt_path)
                self.network.load_state_dict(weights, strict=False)
                self.swa_model.update_parameters(self.network)
                if timestep == self.split_time:
                    break

            # Update BN on last timestep in the train set
            # self.train_dataset.update_current_timestamp(timestep)
            # bn_dataloader = FastDataLoader( 
            #     dataset=self.train_dataset, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            # update_bn(bn_dataloader, self.swa_model, device=self.args.device)


        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def train_offline(self):
        timesteps = self.train_dataset.ENV
        timesteps = self.filter_timestamps(timesteps, online=False)

        for i, t in enumerate(timesteps):
            if t < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif t == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(t)
                train_id_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    pass
                    # self.load_swa_model(t)
                else:
                    self.train_step(train_id_dataloader, self.args.offline_steps)
                    self.swa_model.update_parameters(self.network)
                    if self.update_bn_on_save:
                        tmp_dataloader = FastDataLoader(
                            dataset=self.train_dataset, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                        update_bn(tmp_dataloader, self.swa_model, device=self.args.device)
                    if self.args.swa_ewa:
                        self.swa_model = AveragedModel(self.network, avg_fn=self.ema_avg)
                    else:
                        self.swa_model = AveragedModel(self.network)
                    self.save_swa_model("offline")
                    self.save_model("offline")
                break

    def create_csaw_from_ckpts(self, num_val_timesteps=1, lambda_val=0.50, wise_ft=False, init_ckpt=""):
        timesteps = self.train_dataset.ENV
        
        if self.args.dataset == "fmow":
            self.network.__init__(self.args)
        elif self.args.dataset in ['huffpost', 'arxiv']:
            self.network.__init__(self.train_dataset.num_classes)
        self.network.to(self.args.device)

        self.ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - lambda_val) * averaged_model_parameter + lambda_val * model_parameter
        self.swa_model = AveragedModel(self.network, avg_fn=self.ema_avg)
    
        # Initialize average with the offline checkpoint
        model_path = os.path.join(self.args.results_dir, self.args.exp_path, 'checkpoints', init_ckpt)
        if init_ckpt:
            temp_network = deepcopy(self.network)
            if os.path.exists(os.path.join(model_path, 'time_offline.pth')):
                init_weights = torch.load(os.path.join(model_path, f'time_offline.pth'))
                temp_network.load_state_dict(init_weights, strict=False)
                self.swa_model = AveragedModel(temp_network, avg_fn=ema_avg)
            elif os.path.exists(os.path.join(model_path, 'time_offline_swa.pth')):
                init_weights = torch.load(os.path.join(model_path, f'time_offline_swa.pth'))
                self.swa_model.load_state_dict(init_weights, strict=False)
                self.swa_model.avg_fn = ema_avg

        metrics = []
        for i, t in enumerate(self.train_dataset.ENV):
            # Break on last `num_val_timesteps`
            if timesteps[i + num_val_timesteps] <= self.split_time:
                if wise_ft and t != self.split_time: continue

                ckpt_path = os.path.join(model_path, f'time_{t}.pth')
                weights = torch.load(ckpt_path)
                self.network.load_state_dict(weights, strict=False)
                self.swa_model.update_parameters(self.network)

            if t == self.split_time:
                break
                # self.train_dataset.mode = 0
                # self.train_dataset.update_current_timestamp(t)
                # train_subset = Subset(self.train_dataset, range(1024))
                # print("Update BN")
                # train_dataloader = FastDataLoader(
                #     dataset=train_subset, batch_size=self.eval_batch_size,
                #     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                # update_bn(train_dataloader, self.swa_model, device=self.args.device)

    def find_lambda(self, num_val_timesteps=1, step_size=0.05, min_lambda=0.50):
        # WARNING: NOT COMPATIBLE WITH SHUFFLED INDICES; ASSUME ORDERED DATASETS 
        # Hacky Reinit of Model
        timesteps = self.train_dataset.ENV
        lambda2metric = {}

        for lambda_val in np.arange(min_lambda, 1.01, step_size):
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
                (1 - lambda_val) * averaged_model_parameter + lambda_val * model_parameter

            # Initialize average with the offline checkpoint
            if self.args.dataset == "fmow":
                self.network.__init__(self.args)
            elif self.args.dataset in ['huffpost', 'arxiv']:
                self.network.__init__(self.train_dataset.num_classes)
            self.network.to(self.args.device)

            model_path = os.path.join(self.args.results_dir, self.args.exp_path, 'checkpoints')
            temp_network = deepcopy(self.network)
            if os.path.exists(os.path.join(model_path, 'time_offline.pth')):
                init_weights = torch.load(os.path.join(model_path, f'time_offline.pth'))
                temp_network.load_state_dict(init_weights, strict=False)
                self.swa_model = AveragedModel(temp_network, avg_fn=ema_avg)
            elif os.path.exists(os.path.join(model_path, 'time_offline_swa.pth')):
                init_weights = torch.load(os.path.join(model_path, f'time_offline_swa.pth'))
                self.swa_model.load_state_dict(init_weights, strict=False)
                self.swa_model.avg_fn = ema_avg
            else:
                self.swa_model = AveragedModel(self.network, avg_fn=ema_avg)

            metrics = []
            for i, t in enumerate(timesteps):
                # Break on last `num_val_timesteps`
                if timesteps[i + num_val_timesteps] <= self.split_time:
                    ckpt_path = os.path.join(model_path, f'time_{t}.pth')
                    weights = torch.load(ckpt_path)
                    temp_network.load_state_dict(weights, strict=False)
                    self.swa_model.update_parameters(temp_network)
                elif t <= self.split_time:
                    # self.train_dataset.mode = 0
                    # self.train_dataset.update_current_timestamp(timesteps[i-1])
                    # train_subset = Subset(self.train_dataset, range(2048))
                    # train_dataloader = FastDataLoader(
                    #     dataset=train_subset, batch_size=self.eval_batch_size,
                    #     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                    # update_bn(train_dataloader, self.swa_model, device=self.args.device)
                    self.train_dataset.update_current_timestamp(t)
                    # train_subset = Subset(self.train_dataset, range(2048 * 4 ))
                    train_dataloader = FastDataLoader(
                        dataset=self.train_dataset, batch_size=self.eval_batch_size,
                        num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                    metric = self.network_evaluation(train_dataloader)

                    print(f"Eval on {t} with decay {lambda_val}: {metric}")
                    metrics.append(metric)

                if t == self.split_time: break

            lambda2metric[lambda_val] = np.mean(metrics)
            print(f'Validation Loss with Lambda {round(lambda_val, 2)}: {lambda2metric[lambda_val]}')
        return lambda2metric



    def save_swa_model(self, timestamp):
        backup_state_dict = self.swa_model.state_dict()

        # self.optimizer.swap_swa_sgd()
        swa_model_path = self.get_model_path(str(timestamp) + "_swa")
        torch.save(self.swa_model.state_dict(), swa_model_path)
        self.swa_model.load_state_dict(backup_state_dict, strict=False)

    def load_swa_model(self, timestamp):
        swa_model_path = self.get_model_path(timestamp)
        self.swa_model.load_state_dict(torch.load(swa_model_path), strict=False)

    def train_online(self):
        self.swa_scheduler.step()
        timestamps = self.train_dataset.ENV [:-1]
        timestamps = self.filter_timestamps(timestamps, online=True)

        for i, t in enumerate(timestamps):
            logger.info(f"Training at timestamp {t}")
            if self.args.load_model and self.model_path_exists(t):
                self.load_model(t)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(t)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset, weights=None,
                    batch_size=self.mini_batch_size,
                    num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader, self.args.online_steps)
                logger.info("==== Updating Weights of Averaged Model ====")
                self.swa_model.update_parameters(self.network)
                if self.update_bn_on_save:
                    tmp_dataloader = FastDataLoader(
                        dataset=self.train_dataset, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                    update_bn(tmp_dataloader, self.swa_model, device=self.args.device)
                self.save_model(t)
                self.save_swa_model(t)

                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

            if (self.args.eval_fix or self.args.eval_warmstart_finetune) and t == self.train_split_time:
                break

    def get_swa_model_copy(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa_copy"
        torch.save(self.swa_model, swa_model_path)
        return torch.load(swa_model_path)

    def evaluate_online(self):
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, t in enumerate(self.eval_dataset.ENV[:end]):
            model_checkpoint = copy.deepcopy(self.network)
            self.load_swa_model(t)

            avg_acc, worst_acc, best_acc = self.evaluate_stream(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc

            self.network = model_checkpoint

    def __str__(self):
        return f'SWA-{self.base_trainer_str}'
