# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Modern PyTorch criterion implementations without Fairseq dependencies
from .l2_loss import L2Loss, L2LossWithFlag, BaseCriterion, CriterionOutput
from .l1_loss import L1Loss, L1LossWithFlag
from .binary_logloss import BinaryLogLoss, BinaryLogLossWithFlag
from .multiclass_cross_entropy import MulticlassCrossEntropy, MulticlassCrossEntropyWithFlag
from .mae_deltapos import MAEDeltaPos

# Modern criterion registry - simple dictionary mapping names to classes
CRITERION_REGISTRY = {
    # L2 (MSE) losses
    "l2_loss": L2Loss,
    "l2_loss_with_flag": L2LossWithFlag,
    
    # L1 (MAE) losses
    "l1_loss": L1Loss,
    "l1_loss_with_flag": L1LossWithFlag,
    
    # Binary classification losses
    "binary_logloss": BinaryLogLoss,
    "binary_logloss_with_flag": BinaryLogLossWithFlag,
    
    # Multiclass classification losses
    "multiclass_cross_entropy": MulticlassCrossEntropy,
    "multiclass_cross_entropy_with_flag": MulticlassCrossEntropyWithFlag,
    
    # Special losses for molecular dynamics
    "mae_deltapos": MAEDeltaPos,
}

def get_criterion(name: str):
    """Get a criterion class by name."""
    if name not in CRITERION_REGISTRY:
        raise ValueError(f"Unknown criterion: {name}. Available: {list(CRITERION_REGISTRY.keys())}")
    return CRITERION_REGISTRY[name]

__all__ = [
    "BaseCriterion",
    "CriterionOutput", 
    "L2Loss",
    "L2LossWithFlag",
    "L1Loss", 
    "L1LossWithFlag",
    "BinaryLogLoss",
    "BinaryLogLossWithFlag",
    "MulticlassCrossEntropy",
    "MulticlassCrossEntropyWithFlag",
    "MAEDeltaPos",
    "CRITERION_REGISTRY",
    "get_criterion",
]
