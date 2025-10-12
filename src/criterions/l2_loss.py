# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Dict, override
from .base import BaseCriterion, CriterionOutput


class L2Loss(BaseCriterion):
    """
    Modern PyTorch implementation of L2 loss (MSE) for Graphormer model training.
    Removes Fairseq dependencies while preserving functionality.
    """
    @override
    def __init__(self):
        super().__init__()
        # Normalization constants for molecular dynamics data
        self.target_mean = 6.529300030461668
        self.target_std = 1.9919705951218716
        self.loss_fn = nn.MSELoss(reduction="none")
    
    @override
    def forward(self, model: nn.Module, sample: Dict[str, torch.Tensor], reduce: bool = True) -> CriterionOutput:
        """Compute the L2 loss for the given sample.
        
        Args:
            model: The Graphormer model
            sample: Dictionary containing:
                - "net_input": Model input data including "batched_data"
                - "target": Target values
                - "nsamples": Number of samples in batch
            reduce: Whether to reduce the loss (legacy compatibility)
            
        Returns:
            CriterionOutput with loss, sample_size, and logging info
        """
        sample_size = sample["x"].size(0)
        natoms = sample["x"].size(0)
        
        # Forward pass through model
        logits = model(sample, sample.get('perturb', None))
        
        # Handle sample weight estimation (if model returns weights)
        if isinstance(logits, tuple):
            # only applies when sample weight estimation is used, which is never the case for us
            logits, weights = logits
        else:
            weights = torch.ones_like(logits, dtype=logits.dtype, device=logits.device)
            
        # Get targets from model (maintains compatibility with different target formats)
        targets = sample["y"]
        
        # Normalize targets using molecular dynamics constants
        targets_normalized = (targets - self.target_mean) / self.target_std
        
        # Compute MSE loss with weights
        loss = self.loss_fn(logits, targets_normalized[:logits.size(0)])
        loss = (loss * weights).sum()
        
        logging_output = {
            "loss": loss.detach(),
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        
        return CriterionOutput(
            loss=loss,
            sample_size=sample_size,
            logging_output=logging_output
        )

    @override
    def reduce_metrics(self, logging_outputs: list) -> Dict[str, float]:
        """Aggregate logging outputs from distributed training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        return {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "sample_size": sample_size,
        }
