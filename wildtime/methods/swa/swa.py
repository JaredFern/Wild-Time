import copy
import os

import torch
from ..dataloaders import InfiniteDataLoader
from ..base_trainer import BaseTrainer
# from torchcontrib.optim import SWA as SWA_optimizer
from torch.optim.swa_utils import AveragedModel, SWALR


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

        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def train_offline(self):
        for i, t in enumerate(self.train_dataset.ENV):
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
                    self.load_swa_model(t)
                else:
                    self.train_step(train_id_dataloader, self.args.offline_steps)
                    self.save_swa_model(t)
                break

    def save_swa_model(self, timestamp):
        backup_state_dict = self.swa_model.state_dict()

        # self.optimizer.swap_swa_sgd()
        swa_model_path = self.get_model_path(timestamp) + "_swa"
        torch.save(self.swa_model.state_dict(), swa_model_path)

        self.swa_model.load_state_dict(backup_state_dict)

    def load_swa_model(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa"
        self.swa_model.load_state_dict(torch.load(swa_model_path), strict=False)

    def train_online(self):
        print("==== Updating Weights of Averaged Model ====")
        self.swa_model.update_parameters(self.network)
        self.swa_scheduler.step()

        if len(self.args.eval_fixed_timesteps):
            train_timesteps = self.args.eval_fixed_timesteps
        else:
            train_timesteps = self.train_dataset.ENV[:-1]

        for i, t in enumerate(train_timesteps):
            print(f"Training at timestamp {t}")
            if self.args.load_model and self.model_path_exists(t):
                self.load_model(t)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(t)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader, self.args.online_steps)

                self.save_swa_model(t)
                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

            if (self.args.eval_fix or self.args.eval_warmstart_finetune) and t == self.split_time:
                print("==== Updating Weights of Averaged Model ====")
                self.swa_model.update_parameters(self.network)
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
