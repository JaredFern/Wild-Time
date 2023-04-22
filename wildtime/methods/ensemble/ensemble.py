import os
import torch
from ..dataloaders import InfiniteDataLoader
from ..base_trainer import BaseTrainer


class TemporalEnsemble(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')
        self.model_ensemble = {}

    def train_online(self):
        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            print(f"Training at timestamp {t}")
            model_copy = self.network.detach().clone()

            if self.args.lisa and i == self.args.lisa_start_time:
                self.lisa = True
            self.train_dataset.update_current_timestamp(t)
            if self.args.method in ['simclr', 'swav']:
                self.train_dataset.ssl_training = True

            train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            self.train_step(train_dataloader, self.args.online_steps)

            self.model_ensemble = self.network.detach().clone()
            self.network = model_copy
            if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                self.train_dataset.update_historical(i + 1, data_del=True)

    def forward_pass(self, x, output_features=False):
        features = []
        logits = []

        for timestamp, model in self.model_ensemble.items():
            feature = model.enc(x)
            logit = model.classifier(feature)

            features.append(feature)
            logits.append(logit)



