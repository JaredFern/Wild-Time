from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

IMG_HEIGHT = 224
NUM_CLASSES = 62

class FMoWNetwork(nn.Module):
    def __init__(self, args, weights=None, ssl_training=False):
        super(FMoWNetwork, self).__init__()
        self.args = args
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)

        if weights is not None:
            self.load_state_dict(deepcopy(weights))
        # SimCLR projection head
        if self.args.method == 'simclr':
            from lightly.models.modules.heads import SimCLRProjectionHead
            self.projection_head = SimCLRProjectionHead(1024, 1024, 128)
        # SwaV: projection head and prototypes
        elif self.args.method == 'swav':
            from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
            self.projection_head = SwaVProjectionHead(1024, 1024, 128)
            self.prototypes = SwaVPrototypes(128, n_prototypes=1024)
        elif self.args.time_conditioned:
            self.classifier = nn.Sequential(
                nn.Linear(1024, self.num_classes)
            )
        self.ssl_training = ssl_training

    def freeze_classifier(self):
        for param in self.enc.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.enc.parameters():
            param.requires_grad = True

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))


    def forward(self, x):
        if isinstance(x, list):
            x, timestamp = x
        features = self.enc(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        if self.args.method == 'simclr' and self.ssl_training:
            return self.projection_head(out)
        elif self.args.method == 'swav' and self.ssl_training:
            out = self.projection_head(out)
            out = nn.functional.normalize(out, dim=1, p=2)
            out = self.prototypes(out)
            return out
        elif self.args.time_conditioned:
            out = self.classifier(
                torch.cat([out, torch.unsqueeze(timestamp, dim=1)], dim=1))
            return out
        else:
            return self.classifier(out)

    def forward_features(self, x):
        out = self.enc(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out