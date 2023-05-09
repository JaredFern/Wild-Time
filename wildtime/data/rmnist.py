import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from .utils import download_detection
from torch.utils.data import Dataset


class RMnistBase(Dataset):
    def __init__(self, args):
        super().__init__()

        self.data_file = 'rmnist.pkl'
        self.datasets = np.load(os.path.join(args.data_dir, self.data_file), allow_pickle=True).item()

        self.args = args
        self.shift_stride = args.timestep_stride
        self.num_classes = 10
        self.current_time = 0
        self.resolution = 28
        self.mini_batch_size = args.mini_batch_size
        self.mode = 0
        self.ssl_training = False

        self.ENV = list(sorted(self.datasets[0].keys()))[0:16:args.timestep_stride] # [::args.timestep_stride]
        self.num_tasks = len(self.ENV)

    def update_historical(self, idx, data_del=True):
        try:
            time = self.ENV[idx]
        except:
            import ipdb; ipdb.set_trace()
        prev_time = self.ENV[idx - 1]
        self.datasets[self.mode][time]['images'] = np.concatenate(
            (self.datasets[self.mode][time]['images'], self.datasets[self.mode][prev_time]['images']), axis=0)
        self.datasets[self.mode][time]['labels'] = np.concatenate(
            (self.datasets[self.mode][time]['labels'], self.datasets[self.mode][prev_time]['labels']), axis=0)
        if data_del:
            del self.datasets[self.mode][prev_time]['images']
            del self.datasets[self.mode][prev_time]['labels']

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'rmnist'

class RMnist(RMnistBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        image = self.datasets[self.mode][self.current_time]['images'][index]
        label = self.datasets[self.mode][self.current_time]['labels'][index]
        image_tensor = torch.FloatTensor(image).reshape(self.resolution, self.resolution)
        image_tensor = image_tensor.expand((3, self.resolution, self.resolution))
        label_tensor = torch.argmax(torch.LongTensor([label]))

        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            tensor_to_PIL = transforms.ToPILImage()
            image_tensor = tensor_to_PIL(image_tensor)
            return image_tensor, label_tensor, ''

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.mode][self.current_time]['labels'])


class RMnistGroup(RMnistBase):
    # TODO Implement RMNist Group
    def __init__(self, args):
        super().__init__(args=args)
        pass
        # self.num_groups = args.num_groups
        # self.group_size = args.group_size
        # self.window_end = self.ENV[0]
        # self.groupnum = 0

    def __getitem__(self, index):
        pass
        # if self.mode == 0:
        #     np.random.seed(index)
        #     # Select group ID
        #     idx = self.ENV.index(self.current_time)
        #     if self.args.non_overlapping:
        #         possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
        #         if len(possible_groupids) == 0:
        #             possible_groupids = [np.random.randint(self.group_size)]
        #     else:
        #         possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
        #     groupid = np.random.choice(possible_groupids)

        #     # Pick a time step in the sliding window
        #     window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
        #     sel_time = self.ENV[np.random.choice(window)]
        #     start_idx, end_idx = self.task_idxs[sel_time][self.mode]

        #     # Pick an example in the time step
        #     sel_idx = np.random.choice(np.arange(start_idx, end_idx))
        #     image = self.datasets[self.current_time][self.mode]['images'][sel_idx]
        #     label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]

        #     image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        #     label_tensor = torch.LongTensor([label])
        #     group_tensor = torch.LongTensor([groupid])

        #     del groupid
        #     del window
        #     del sel_time
        #     del start_idx
        #     del end_idx
        #     del sel_idx

        #     return image_tensor, label_tensor, group_tensor

        # else:
        #     image = self.datasets[self.current_time][self.mode]['images'][index]
        #     label = self.datasets[self.current_time][self.mode]['labels'][index]
        #     image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        #     label_tensor = torch.LongTensor([label])

        #     return image_tensor, label_tensor

    def group_counts(self):
        pass
        # idx = self.ENV.index(self.current_time)
        # return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        pass
        # return len(self.datasets[self.current_time][self.mode]['labels'])
