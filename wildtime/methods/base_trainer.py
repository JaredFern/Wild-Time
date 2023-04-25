import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from tdc import Evaluator
from tqdm import tqdm

from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import prepare_data, forward_pass, get_collate_functions, reinit_dataset

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = 2
        self.num_classes = dataset.num_classes
        self.num_tasks = dataset.num_tasks
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(args, self.train_dataset)

        # Training hyperparameters
        self.args = args
        self.train_update_iter = args.train_update_iter
        self.lisa = args.lisa
        self.mixup = args.mixup
        self.cut_mix = args.cut_mix
        self.mix_alpha = args.mix_alpha
        self.mini_batch_size = args.mini_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.num_workers = args.num_workers
        self.checkpoint_path = args.checkpoint_path
        self.split_time = args.split_time
        self.eval_next_timestamps = args.eval_next_timestamps
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = 'accuracy'
        if str(self.eval_dataset) == 'drug':
            self.eval_metric = 'PCC'
        elif 'mimic' in str(self.eval_dataset) and self.args.prediction_type == 'mortality':
            self.eval_metric = 'ROC-AUC'

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'train_update_iter={self.train_update_iter}-lr={self.args.lr}-' \
                                f'mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}'
        if self.args.lisa:
            base_trainer_str += f'-lisa-mix_alpha={self.mix_alpha}'
        elif self.mixup:
            base_trainer_str += f'-mixup-mix_alpha={self.mix_alpha}'
        if self.cut_mix:
            base_trainer_str += f'-cut_mix'
        if self.args.eval_fix:
            base_trainer_str += f'-eval_fix'
        else:
            base_trainer_str += f'-eval_stream'
        return base_trainer_str

    def train_step(self, dataloader, num_steps=None):
        if num_steps == None:
            num_steps = self.args.train_update_iter
        self.network.train()
        loss_all = []
        progress_bar = tqdm(total=num_steps)

        if self.args.linear_probe_iter:
            logger.info("==== Freezing Encoder for Linear Probing ====")
            self.network.freeze_classifier()

        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(
                x, y, self.train_dataset, self.network,
                self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step > 0 and step == self.args.linear_probe_iter:
                logger.info("==== Unfreezing Encoder for full Finetuning ====")
                self.network.unfreeze_classifier()
            if self.args.method in ['swa'] and \
                    self.args.swa_steps is not None and step % self.args.swa_steps == 0:
                logger.info("==== Updating Weights of Averaged Model ====")
                self.swa_model.update_parameters(self.network)

            if step % 50 == 0:
                logger.info(f'Train Loss at step {step}: {loss.item()}')
                # logger.info(f'LR: {self.swa_scheduler.get_lr()}')

            progress_bar.update(1)
            if step == num_steps:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_online(self):
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            logger.info(f"Training at timestamp: {timestamp}")
            if self.args.load_model and self.model_path_exists(timestamp):
                self.load_model(timestamp)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader, self.args.online_steps)
                self.save_model(timestamp)
                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)
            if (self.args.eval_fix or self.args.eval_warmstart_finetune) and timestamp == self.split_time:
                break

    def train_offline(self):
        if self.args.method in ['simclr', 'swav']:
            self.train_dataset.ssl_training = True

        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(timestamp)
                else:
                    self.train_step(train_id_dataloader, self.args.offline_steps)
                    self.save_model("offline")
                break

    def train_fixed_timesteps(self):
        for timestep in self.args.eval_fixed_timesteps:
            self.train_dataset.update_current_timestamp(timestep)
            train_dataloader = InfiniteDataLoader(
                dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            self.train_step(train_dataloader, 500)
            if self.args.method in ['swa']:
                logger.info("==== Updating Weights of Averaged Model ====")
                self.swa_model.update_parameters(self.network)

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                if self.args.method in ['swa']:
                    logits = self.swa_model(x)
                else:
                    logits = self.network(x)

                if self.args.dataset in ['drug']:
                    pred = logits.reshape(-1, )
                else:
                    pred = F.softmax(logits, dim=1).argmax(dim=1)

                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()

        if self.args.dataset == 'drug':
            evaluator = Evaluator(name='PCC')
            metric = evaluator(y_all, pred_all)
        else:
            pred_all = np.array(pred_all)
            y_all = np.array(y_all)
            if self.args.dataset == 'mimic' and self.args.prediction_type == 'mortality':
                metric = metrics.roc_auc_score(y_all, pred_all)
            else:
                correct = (pred_all == y_all).sum().item()
                metric = correct / float(y_all.shape[0])
        self.network.train()

        return metric

    def network_featurize(self, test_time_dataloader):
        self.network.eval()
        features_all = []
        preds_all = []
        labels_all = []

        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))

            with torch.no_grad():
                features = self.network.forward_features(x).detach()
                predictions = self.network.classifier(features).detach()

                features_all.append(features)
                preds_all.append(predictions)
                labels_all.append(y.detach())

        features_all = torch.cat(features_all, dim=0)
        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        return features_all, preds_all, labels_all


    def evaluate_stream(self, start):
        self.network.eval()
        metrics = []
        for i in range(start, min(start + self.eval_next_timestamps, len(self.eval_dataset.ENV))):
            test_time = self.eval_dataset.ENV[i]
            self.eval_dataset.update_current_timestamp(test_time)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.eval_batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric = self.network_evaluation(test_time_dataloader)
            metrics.append(metric)

        avg_metric, worst_metric, best_metric = np.mean(metrics), np.min(metrics), np.max(metrics)

        logger.info(
            f'Timestamp = {start - 1}'
            f'\t Average {self.eval_metric}: {avg_metric}'
            f'\t Worst {self.eval_metric}: {worst_metric}'
            f'\t Best {self.eval_metric}: {best_metric}'
            f'\t Performance over all timestamps: {metrics}\n'
        )
        self.network.train()

        return avg_metric, worst_metric, best_metric

    def extract_features(self):
        logger.info(f'\n=================================== Results (Eval-Features) ===================================')
        self.network.eval()
        features_by_time, preds_by_time, labels_by_time, projections_by_time, timestamps = {}, {}, {}, {}, {}

        for i, _ in enumerate(self.eval_dataset.ENV):
            self.eval_dataset.update_current_timestamp(i)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.eval_batch_size,
                                                num_workers=self.num_workers, collate_fn=self.eval_collate_fn)

            features, preds, labels = self.network_featurize(test_time_dataloader)

            features_by_time[i] = features
            preds_by_time[i] = preds
            labels_by_time[i] = labels
            timestamps[i] = i * torch.ones_like(labels)
            U, S, V = torch.pca_lowrank(features, q=self.args.feat_num_components)
            projections_by_time[i] = features @ V

        timestamps_all = torch.cat(list(timestamps.values()), dim=0)
        features_all = torch.cat(list(features_by_time.values()), dim=0)
        U_all, S_all, V_all = torch.pca_lowrank(features_all, q=self.args.feat_num_components)
        projections_all = features_all @ V_all

    def evaluate_online(self):
        logger.info(f'\n=================================== Results (Eval-Stream) ===================================')
        logger.info(f'Metric: {self.eval_metric}\n')
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, timestamp in enumerate(self.eval_dataset.ENV[:end]):
            self.load_model(timestamp)
            avg_metric, worst_metric, best_metric = self.evaluate_stream(i + 1)
            self.task_accuracies[timestamp] = avg_metric
            self.worst_time_accuracies[timestamp] = worst_metric
            self.best_time_accuracies[timestamp] = best_metric

    def evaluate_offline(self):
        logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.eval_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric = self.network_evaluation(test_id_dataloader)
                logger.info(f'ID {self.eval_metric}: \t{id_metric}\n')
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.eval_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                logger.info(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {acc}')
                metrics.append(acc)
        logger.info(f'\nOOD Average Metric: \t{np.mean(metrics)}'
              f'\nOOD Worst Metric: \t{np.min(metrics)}'
              f'\nAll OOD Metrics: \t{metrics}\n')

    def evaluate_offline_all_timestamps(self):
        logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        timestamps = self.train_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp <= self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.eval_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric = self.network_evaluation(test_id_dataloader)
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.eval_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric = self.network_evaluation(test_ood_dataloader)
            logger.info(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
            metrics.append(metric)
        logger.info(f'\nAverage Metric Across All Timestamps: \t{np.mean(metrics)}'
              f'\nWorst Metric Across All Timestamps: \t{np.min(metrics)}'
              f'\nMetrics Across All Timestamps: \t{metrics}\n')

    def run_eval_fix(self):
        logger.info('==========================================================================================')
        logger.info("Running Eval-Fix...\n")
        if self.args.method in ['agem', 'ewc', 'ft', 'si']:
            self.train_online()
        else:
            self.train_offline()
        if self.args.eval_all_timestamps:
            self.evaluate_offline_all_timestamps()
        else:
            self.evaluate_offline()

    def run_task_difficulty(self):
        logger.info('==========================================================================================')
        logger.info("Running Task Difficulty...\n")
        timestamps = self.train_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            self.train_dataset.mode = 0
            self.train_dataset.update_current_timestamp(timestamp)
            if i < len(timestamps) - 1:
                self.train_dataset.update_historical(i + 1)
            else:
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(timestamp)
                else:
                    self.train_step(train_id_dataloader)
                    self.save_model(timestamp)

        for i, timestamp in enumerate(timestamps):
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(timestamp)
            test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                 batch_size=self.eval_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric = round(self.network_evaluation(test_ood_dataloader), 2)
            logger.info(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
            metrics.append(metric)
        logger.info(f'Average Metric: {np.mean(metrics)}')
        logger.info(f'Worst timestamp accuracy: {np.min(metrics)}')
        logger.info(f'All timestamp accuracies: {metrics}')

    def run_eval_stream(self):
        logger.info('==========================================================================================')
        logger.info("Running Eval-Stream...\n")
        if not self.args.load_model:
            self.train_online()
        self.evaluate_online()


    def run_warmstart_finetune(self):
        # Initialize with general supervised training
        self.train_offline()

       ## Reset to online dataset
        self.train_dataset = reinit_dataset(self.args)

        # Continual train on each timestamp
        self.train_online()
        self.evaluate_offline()

    def run_train_fixed_timesteps(self):
        self.train_fixed_timesteps()
        self.evaluate_offline()

    def run_eval_features(self):
        self.train_offline()
        self.extract_features()

    def run(self):
        torch.cuda.empty_cache()
        start_time = time.time()

        if self.args.difficulty:
            self.run_task_difficulty()
        elif self.args.eval_fix:
            self.run_eval_fix()
        elif self.args.eval_warmstart_finetune:
            self.run_warmstart_finetune()
        elif self.args.eval_features:
            self.run_eval_features()
        elif self.args.eval_fixed_timesteps:
            self.run_train_fixed_timesteps()
        else:
            self.run_eval_stream()

        runtime = time.time() - start_time
        logger.info(f'Runtime: {runtime:.2f}\n')

    def get_model_path(self, timestamp):
        model_str = f'time_{timestamp}.pth'
        path = os.path.join(self.args.model_path, model_str)
        return path

    def model_path_exists(self, timestamp):
        return os.path.exists(self.get_model_path(timestamp))

    def save_model(self, timestamp):
        path = self.get_model_path(timestamp)
        torch.save(self.network.state_dict(), path)
        logger.info(f'Saving model at timestamp {timestamp} to path {path}...\n')

    def load_model(self, timestamp, checkpoint_path=None):
        if self.checkpoint_path:
            path = self.checkpoint_path
        else:
            path = self.get_model_path(timestamp)
        self.network.load_state_dict(torch.load(path), strict=False)
