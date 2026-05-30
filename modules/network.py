"""Definition of the NN architecture"""

import torch
import torch.nn as nn
from torchvision import models

class UnimodalNet(nn.Module):
    def __init__(self, task, unfreeze_last_block=True):
        """
        task : 'realfake' (2 classes) ou 'transform' (3 classes)
        """
        super().__init__()
        assert task in ("realfake", "transform")
        self.task = task
        n_classes = 2 if task == "realfake" else 3

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features      # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        feats = self.backbone(x)        # (batch, 512)
        return self.head(feats)         # (batch, n_classes)

class JointDetectNet(nn.Module):
    def __init__(self, n_transforms=3, unfreeze_last_block=True):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features         
        backbone.fc = nn.Identity()                  # on retire la tête de classif ImageNet
        self.backbone = backbone

        # 1. tout geler
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2. dégeler seulement le dernier bloc (layer4)
        if unfreeze_last_block:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        # deux têtes indépendantes sur les mêmes features
        self.head_realfake = nn.Linear(feat_dim, 2)            # real / ai
        self.head_transform = nn.Linear(feat_dim, n_transforms) # original / transfer / redigital

    def forward(self, x):
        feats = self.backbone(x)                # (batch, 512)
        return self.head_realfake(feats), self.head_transform(feats)