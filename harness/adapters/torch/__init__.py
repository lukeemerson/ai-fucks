"""TorchVision-backed adapters for the harness.

These adapters are gated behind the ``[experiment]`` optional-dependency
group; importing this package requires ``torch`` and ``torchvision``. The
TXRV (NIH-pretrained) backbone additionally requires ``torchxrayvision``.
"""

from harness.adapters.torch.backbone import (
    TorchVisionDenseNet121Backbone,
    TorchVisionResNet50Backbone,
)
from harness.adapters.torch.txrv_backbone import TXRVDenseNet121NIHBackbone

__all__ = [
    "TXRVDenseNet121NIHBackbone",
    "TorchVisionDenseNet121Backbone",
    "TorchVisionResNet50Backbone",
]
