# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .l2_loss import BaseCriterion, CriterionOutput


class L1Loss(BaseCriterion):
    """
    Modern PyTorch implementation of L1 loss (MAE) for Graphormer model training.
    Removes Fairseq dependencies while preserving functionality.
    """
    
    def __init__(self):
        super().__init__()
        # Normalization constants for molecular dynamics data
        self.target_mean = 6.529300030461668
        self.target_std = 1.9919705951218716
        
    def forward(self, model, sample: Dict[str, Any], reduce: bool = True) -> CriterionOutput:
        """Compute the L1 loss for the given sample.
        
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
        sample_size = sample["nsamples"]
        
        # Get number of atoms for logging
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]
        
        # Forward pass through model
        logits = model(**sample["net_input"])
        
        # Handle sample weight estimation (if model returns weights)
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)
            
        # Get targets from model (maintains compatibility with different target formats)
        targets = model.get_targets(sample, [logits])
        
        # Normalize targets using molecular dynamics constants
        targets_normalized = (targets - self.target_mean) / self.target_std
        
        # Compute MAE loss with weights
        loss = nn.L1Loss(reduction="none")(logits, targets_normalized[:logits.size(0)])
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
    
    def reduce_metrics(self, logging_outputs: list) -> Dict[str, float]:
        """Aggregate logging outputs from distributed training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        return {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "sample_size": sample_size,
        }


class L1LossWithFlag(L1Loss):
    """
    L1 loss with FLAG adversarial training support.
    Modern PyTorch implementation without Fairseq dependencies.
    """
    
    def forward(self, model, sample: Dict[str, Any], reduce: bool = True) -> CriterionOutput:
        """Compute the L1 loss with FLAG perturbations.
        
        Args:
            model: The Graphormer model
            sample: Dictionary containing input data and targets
            reduce: Whether to reduce the loss (legacy compatibility)
            
        Returns:
            CriterionOutput with loss, sample_size, and logging info
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)  # FLAG perturbations
        
        # Get number of atoms for logging
        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
            
        # Forward pass with perturbations
        logits = model(**sample["net_input"], perturb=perturb)
        
        # Handle sample weight estimation
        if isinstance(logits, tuple):
            logits, weights = logits
        else:
            weights = torch.ones(logits.shape, dtype=logits.dtype, device=logits.device)
            
        # Get and normalize targets
        targets = model.get_targets(sample, [logits])
        targets_normalized = (targets - self.target_mean) / self.target_std
        
        # Compute weighted MAE loss
        loss = nn.L1Loss(reduction="none")(logits, targets_normalized[:logits.size(0)])
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