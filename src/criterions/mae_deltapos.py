# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

##########################################
# !! NOT FULLY CHECKED BECAUSE UNUSED !! #
##########################################

from typing import Dict, Any, override
import torch
import torch.nn as nn
from .base import BaseCriterion, CriterionOutput


class MAEDeltaPos(BaseCriterion):
    """
    Modern PyTorch implementation of MAE DeltaPos loss for IS2RE task.
    Removes Fairseq dependencies while preserving functionality.
    Handles energy and position prediction for molecular structures.
    """
    
    # Energy and deltapos normalization constants
    e_thresh = 0.02
    e_mean = -1.4729953244844094
    e_std = 2.2707848125378405
    d_mean = [0.1353900283575058, 0.06877671927213669, 0.08111362904310226]
    d_std = [1.7862379550933838, 1.78688645362854, 0.8023099899291992]
    
    @override
    def __init__(self, node_loss_weight: float = 1.0, min_node_loss_weight: float = 0.01, max_update: int = 1000):
        super().__init__()
        self.node_loss_weight = node_loss_weight
        self.min_node_loss_weight = min_node_loss_weight
        self.max_update = max_update
        self.node_loss_weight_range = max(0, self.node_loss_weight - self.min_node_loss_weight)
        self.num_updates = 0
        self.loss_fn = nn.L1Loss(reduction="none")
        
    def _set_num_updates(self, num_updates: int):
        """Set the current number of updates for weight scheduling."""
        self.num_updates = num_updates
    
    @override
    def forward(self, model, sample: Dict[str, Any], reduce: bool = True) -> CriterionOutput:
        """Compute the MAE deltapos loss for the given sample.
        
        Args:
            model: The Graphormer model
            sample: Dictionary containing:
                - "net_input": Model input data including "batched_data" and "atoms"
                - "targets": Target values including "relaxed_energy" and "deltapos"
                - "nsamples": Number of samples in batch
            reduce: Whether to reduce the loss (legacy compatibility)
            
        Returns:
            CriterionOutput with loss, sample_size, and logging info
        """
        # Calculate dynamic node loss weight based on training progress
        node_loss_weight = (
            self.node_loss_weight
            - self.node_loss_weight_range * self.num_updates / self.max_update
        )
        
        # Count valid nodes
        valid_nodes = sample["atoms"].ne(0).sum()
        
        # Forward pass through model
        output, node_output, node_target_mask = model(**sample["net_input"])
        
        # Process energy targets
        relaxed_energy = sample["targets"]["relaxed_energy"].float()
        relaxed_energy = (relaxed_energy - self.e_mean) / self.e_std
        sample_size = relaxed_energy.numel()
        
        # Compute energy loss
        energy_loss = self.loss_fn(output.float().view(-1), relaxed_energy)
        with torch.no_grad():
            energy_within_threshold = (energy_loss.detach() * self.e_std < self.e_thresh).sum()
        energy_loss = energy_loss.sum()
        
        # Process deltapos targets
        deltapos = sample["targets"]["deltapos"].float()
        deltapos = (deltapos - deltapos.new_tensor(self.d_mean)) / deltapos.new_tensor(self.d_std)
        deltapos *= node_target_mask
        node_output *= node_target_mask
        target_cnt = node_target_mask.sum(dim=[1, 2])
        
        # Compute node position loss
        node_loss = (
            self.loss_fn(node_output.float(), deltapos)
            .mean(dim=-1)
            .sum(dim=-1)
            / target_cnt
        ).sum()
        
        # Total loss combines energy and position losses
        total_loss = energy_loss + node_loss_weight * node_loss
        
        logging_output = {
            "loss": energy_loss.detach(),
            "energy_within_threshold": energy_within_threshold,
            "node_loss": node_loss.detach(),
            "sample_size": sample_size,
            "nsentences": sample_size,
            "num_nodes": valid_nodes.detach(),
            "node_loss_weight": node_loss_weight * sample_size,
        }
        
        return CriterionOutput(
            loss=total_loss,
            sample_size=sample_size,
            logging_output=logging_output
        )
    
    @override
    def reduce_metrics(self, logging_outputs: list) -> Dict[str, float]:
        """Aggregate logging outputs from distributed training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        energy_within_threshold_sum = sum(log.get("energy_within_threshold", 0) for log in logging_outputs)
        node_loss_sum = sum(log.get("node_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        # Calculate means with proper denormalization
        mean_loss = (loss_sum / sample_size) * self.e_std if sample_size > 0 else 0.0
        energy_within_threshold = energy_within_threshold_sum / sample_size if sample_size > 0 else 0.0
        mean_node_loss = (node_loss_sum / sample_size) * sum(self.d_std) / 3.0 if sample_size > 0 else 0.0
        mean_n_nodes = sum(log.get("num_nodes", 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.0
        node_loss_weight = sum(log.get("node_loss_weight", 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.0
        
        return {
            "loss": mean_loss,
            "ewth": energy_within_threshold,  # Energy within threshold
            "node_loss": mean_node_loss,
            "nodes_per_graph": mean_n_nodes,
            "node_loss_weight": node_loss_weight,
            "sample_size": sample_size,
        }
