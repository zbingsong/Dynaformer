# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Modern PyTorch criterion implementations without Fairseq dependencies
from .base import BaseCriterion, CriterionOutput
from .l2_loss import L2Loss
from .l1_loss import L1Loss
from .binary_logloss import BinaryLogLoss
from .multiclass_cross_entropy import MulticlassCrossEntropy
from .mae_deltapos import MAEDeltaPos

# Modern criterion registry - simple dictionary mapping names to classes
CRITERION_REGISTRY = {
    # L2 (MSE) losses
    "l2_loss": L2Loss,
    # L1 (MAE) losses
    "l1_loss": L1Loss,
    # Binary classification losses
    "binary_logloss": BinaryLogLoss,
    # Multiclass classification losses
    "multiclass_cross_entropy": MulticlassCrossEntropy,
    # Special losses for molecular dynamics
    "mae_deltapos": MAEDeltaPos,
}

def get_criterion(name: str) -> BaseCriterion:
    """Get a criterion class by name."""
    if name not in CRITERION_REGISTRY:
        raise ValueError(f"Unknown criterion: {name}. Available: {list(CRITERION_REGISTRY.keys())}")
    return CRITERION_REGISTRY[name]

__all__ = [
    "BaseCriterion",
    "CriterionOutput", 
    "L2Loss",
    "L1Loss", 
    "BinaryLogLoss",
    "MulticlassCrossEntropy",
    "MAEDeltaPos",
    "CRITERION_REGISTRY",
    "get_criterion",
]
