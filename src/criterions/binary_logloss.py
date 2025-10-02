# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from .l2_loss import BaseCriterion, CriterionOutput


class BinaryLogLoss(BaseCriterion):
    """
    Modern PyTorch implementation of Binary Log Loss for Graphormer model training.
    Removes Fairseq dependencies while preserving functionality.
    Handles NaN target masking and logits indexing.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, model, sample: Dict[str, Any], reduce: bool = True) -> CriterionOutput:
        """Compute the binary log loss for the given sample.
        
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
        
        # Extract the first dimension slice as done in original
        logits = logits[:, 0, :]
        
        # Get targets from model (maintains compatibility with different target formats)
        targets = model.get_targets(sample, [logits])
        
        # Compute predictions for accuracy metrics
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        
        # Flatten tensors and handle NaN masking
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[:logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        
        # Compute binary cross entropy loss with NaN masking
        loss = F.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), 
            targets_flatten[mask].float(), 
            reduction="sum"
        )
        
        logging_output = {
            "loss": loss.detach(),
            "sample_size": torch.sum(mask.type(torch.int64)),
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": (preds == targets[:preds.size(0)]).sum(),
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
        
        metrics = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "sample_size": sample_size,
        }
        
        # Add accuracy if available
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics["accuracy"] = 100.0 * ncorrect / sample_size if sample_size > 0 else 0.0
            
        return metrics


class BinaryLogLossWithFlag(BinaryLogLoss):
    """
    Binary log loss with FLAG adversarial training support.
    Modern PyTorch implementation without Fairseq dependencies.
    """
    
    def forward(self, model, sample: Dict[str, Any], reduce: bool = True) -> CriterionOutput:
        """Compute the binary log loss with FLAG perturbations.
        
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
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        
        # Get targets and compute predictions
        targets = model.get_targets(sample, [logits])
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        
        # Flatten tensors and handle NaN masking
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[:logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        
        # Compute binary cross entropy loss with NaN masking
        loss = F.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), 
            targets_flatten[mask].float(), 
            reduction="sum"
        )
        
        logging_output = {
            "loss": loss.detach(),
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": (preds == targets[:preds.size(0)]).sum(),
        }
        
        return CriterionOutput(
            loss=loss,
            sample_size=sample_size,
            logging_output=logging_output
        )
