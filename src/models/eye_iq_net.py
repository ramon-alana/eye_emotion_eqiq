from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torchvision import models


class EyeIQNet(nn.Module):
    """ResNet18 backbone with multi-task heads for emotion + proxies."""

    def __init__(
        self,
        num_emotions: int,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(in_features, num_emotions)
        self.valence_head = nn.Linear(in_features, 1)
        self.arousal_head = nn.Linear(in_features, 1)
        self.iq_head = nn.Linear(in_features, 1)
        self.eq_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.dropout(features)
        return {
            "emotion_logits": self.emotion_head(features),
            "valence": self.valence_head(features).squeeze(-1),
            "arousal": self.arousal_head(features).squeeze(-1),
            "iq_proxy": self.iq_head(features).squeeze(-1),
            "eq_proxy": self.eq_head(features).squeeze(-1),
        }

    def freeze_backbone(self, except_layers: Optional[int] = None) -> None:
        """Optionally freeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        if except_layers:
            # Unfreeze the last `except_layers` layers in layer4
            for param in list(self.backbone.layer4.parameters())[-except_layers:]:
                param.requires_grad = True



