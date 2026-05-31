"""Definition of the NN architecture"""

import torch
import torch.nn as nn
from torchvision import models

class UnimodalNet(nn.Module):
    def __init__(self, task, unfreeze_last_block=True):
        """
        Unimodal Net with only one classification head trained on one objective only
        specify the task: 'realfake' (binary) or 'transform' (three classes)
        """

        super().__init__()
        assert task in ("realfake", "transform") # check if task is valid

        self.task = task
        n_classes = 2 if task == "realfake" else 3

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features # 512
        backbone.fc = nn.Identity() # Remove the original classification head
        self.backbone = backbone

        # We FREEZE the original backbone, except the last part of the network!
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        self.head = nn.Linear(feat_dim, n_classes) # Add the untrained classification head

    def forward(self, x):
        feats = self.backbone(x) 
        return self.head(feats) # (batch, n_classes)

class JointDetectNet(nn.Module):
    """Network with two independant classification heads added on the backbone.
        Both impact is weighted using the alpha/beta parameters.
        Both head share the same weights from the backbone"""
    
    def __init__(self, n_transforms=3, unfreeze_last_block=True):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features         
        backbone.fc = nn.Identity() 
        self.backbone = backbone

        for p in self.backbone.parameters(): # Same partial freezing
            p.requires_grad = False
        if unfreeze_last_block:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        # Two independant heads
        self.head_realfake = nn.Linear(feat_dim, 2) # real / ai
        self.head_transform = nn.Linear(feat_dim, n_transforms) # original / transfer / redigital

    def forward(self, x):
        feats = self.backbone(x) 
        return self.head_realfake(feats), self.head_transform(feats) 