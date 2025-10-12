from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn


@dataclass
class CriterionOutput:
    """Output from criterion computation."""
    loss: torch.Tensor
    sample_size: int
    logging_output: Dict[str, Any]


class BaseCriterion(nn.Module):
    """Base class for modern PyTorch criterions without Fairseq dependencies."""
    
    def __init__(self):
        super().__init__()

    def forward(self, model: nn.Module, sample: Dict[str, torch.Tensor], reduce: bool = True) -> CriterionOutput:
        """Compute the loss for the given sample."""
        raise NotImplementedError
        
    def reduce_metrics(self, logging_outputs: list) -> Dict[str, float]:
        """Aggregate logging outputs from distributed training."""
        raise NotImplementedError
