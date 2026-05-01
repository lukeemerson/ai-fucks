"""TorchVision-backed adapters for the harness.

These adapters are gated behind the ``[experiment]`` optional-dependency
group; importing this package requires ``torch`` and ``torchvision``.
"""

from harness.adapters.torch.backbone import (
    TorchVisionDenseNet121Backbone,
    TorchVisionResNet50Backbone,
)

__all__ = [
    "TorchVisionDenseNet121Backbone",
    "TorchVisionResNet50Backbone",
]
