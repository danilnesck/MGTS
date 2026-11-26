"""Frozen ImageNet features for training a small condition classifier."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet18

BACKBONES = {
    "mobilenet_v3_small": (mobilenet_v3_small, 576, "mobilenet_v3_small-047dcff4.pth"),
    "efficientnet_b0": (efficientnet_b0, 1280, "efficientnet_b0_rwightman-7f5810bc.pth"),
    "resnet18": (resnet18, 512, "resnet18-f37072fd.pth"),
}


class FeatureClassifier(nn.Module):
    def __init__(self, backbone_name, grids, num_classes, initial=None):
        super().__init__()
        builder, channels, _ = BACKBONES[backbone_name]
        backbone = builder(weights=None)
        if initial is not None:
            backbone.load_state_dict(initial)
        if backbone_name == "resnet18":
            self.features = nn.Sequential(*list(backbone.children())[:-2])
        else:
            self.features = backbone.features
        self.grids = grids
        self.head = nn.Linear(channels * sum(size ** 2 for size in grids), num_classes)

    def embed(self, images):
        patches = self.features(images)
        parts = [
            F.normalize(F.adaptive_avg_pool2d(patches, size).flatten(1)) for size in self.grids
        ]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1) / len(parts) ** 0.5

    def forward(self, images):
        return self.head(self.embed(images))
