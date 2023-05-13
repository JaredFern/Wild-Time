import copy
import logging
import math
import os

import torch
from tqdm import tqdm
from ..dataloaders import FastDataLoader,InfiniteDataLoader
from ..base_trainer import BaseTrainer
# from torchcontrib.optim import SWA as SWA_optimizer
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

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
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.network = self.network
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=args.lr, anneal_epochs=1)

        if args.swa_ewa:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
                (1 - args.swa_ewa_lambda) * averaged_model_parameter + args.swa_ewa_lambda * model_parameter
            self.swa_model = AveragedModel(self.network, avg_fn=ema_avg)
        else:
            self.swa_model = AveragedModel(self.network)

        if args.swa_load_from_checkpoint:
            model_path = os.path.join(self.args.results_dir, self.args.exp_path, 'checkpoints')
            logger.info(f"Building SWA Models from {model_path}")
            init_weights = torch.load(os.path.join(model_path, f'time_{self.train_dataset[ENV][0]}.pth'))
            self.network.load_state_dict(init_weights)

            self.swa_model = AveragedModel(self.network, avg_fn=ema_avg)
            self.swa_model.update_parameters(self.network)

            for timestep in tqdm(self.train_dataset.ENV[1:]):
                ckpt_path = os.path.join(model_path, f'time_{timestep}.pth')
                weights = torch.load(ckpt_path)
                self.network.load_state_dict(weights)
                self.swa_model.update_parameters(self.network)
                if timestep == self.split_time:
                    break


        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def train_offline(self):
        timesteps = self.train_dataset.ENV
        timesteps = self.filter_timestamps(timesteps)


        for i, t in enumerate(timesteps):
            if t < self.split_time: # , self.args.warmstart_split_time):
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif t == self.split_time: #  or t == self.args.warmstart_split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(t)
                train_id_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_swa_model(t)
                else:
                    self.train_step(train_id_dataloader, self.args.offline_steps)
                    # update_bn(train_id_dataloader, self.swa_model, device=self.args.device)
                    self.save_swa_model("offline")
                break

    def save_swa_model(self, timestamp):
        backup_state_dict = self.swa_model.state_dict()

        # self.optimizer.swap_swa_sgd()
        swa_model_path = self.get_model_path(str(timestamp) + "_swa")
        torch.save(self.swa_model.state_dict(), swa_model_path)
        self.swa_model.load_state_dict(backup_state_dict)

    def load_swa_model(self, timestamp):
        swa_model_path = self.get_model_path(timestamp)
        self.swa_model.load_state_dict(torch.load(swa_model_path), strict=False)

    def train_online(self):
        self.swa_scheduler.step()
        timestamps = self.train_dataset.ENV [:-1]
        timestamps = self.filter_timestamps(timestamps)

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
                self.save_model(t)
                self.save_swa_model(t)

                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

            if (self.args.eval_fix or self.args.eval_warmstart_finetune) and t == self.split_time:
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
