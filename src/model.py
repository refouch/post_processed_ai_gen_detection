"""
Multi-task detector: shared EfficientNet-B0 backbone + two classification heads.

  real_fake_head : binary (real=0, ai=1)
  transform_head : 3-class (original=0, transfer=1, redigital=2)

The forward pass always returns both logits; at inference time you can
mask out the head you don't need with task_weight = 0 in the loss.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class MultiTaskDetector(nn.Module):
    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Keep everything except the original classifier
        self.backbone = base.features          # outputs (B, 1280, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)    # -> (B, 1280, 1, 1)
        in_features = 1280

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.real_fake_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2),
        )
        self.transform_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(self.backbone(x)).flatten(1)
        return self.real_fake_head(features), self.transform_head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features (useful for analysis)."""
        return self.pool(self.backbone(x)).flatten(1)


def multitask_loss(
    rf_logits: torch.Tensor,
    tr_logits: torch.Tensor,
    rf_labels: torch.Tensor,
    tr_labels: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for joint training.

      total = alpha * CE(real_fake) + (1 - alpha) * CE(transform)

    alpha = 1.0 → real/fake baseline only
    alpha = 0.0 → transform baseline only
    alpha = 0.5 → equal weighting

    Returns (total_loss, rf_loss, tr_loss).
    """
    rf_loss = nn.functional.cross_entropy(rf_logits, rf_labels)
    tr_loss = nn.functional.cross_entropy(tr_logits, tr_labels)
    total = alpha * rf_loss + (1.0 - alpha) * tr_loss
    return total, rf_loss, tr_loss
