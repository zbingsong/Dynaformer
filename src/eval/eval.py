import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, Union


class Evaluator:
    def __init__(
            self, 
            model: nn.Module, 
            device: str="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        # self.target_mean = 6.529300030461668
        # self.target_std = 1.9919705951218716
        self.device = device


    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for sample in dataloader:
                sample = self._move_to_device(sample)
                logits: torch.Tensor = self.model(sample).detach() # logits shape: [batch, num_classes]
                all_logits.append(logits)

                targets = sample["y"]
                # Normalize targets using molecular dynamics constants
                # targets_normalized = (targets - self.target_mean) / self.target_std
                all_targets.append(targets)

        logits = torch.cat(all_logits, dim=0).squeeze(1) # shape: [total_samples, num_classes]
        targets = torch.cat(all_targets, dim=0) # shape: [total_samples]
        # Compute metrics
        mse_loss = F.mse_loss(logits, targets, reduction='mean')
        pearson_r = torch.corrcoef(torch.stack([logits, targets]))[0,1]
        cindex = self._get_cindex(logits, targets)

        return {
            "mse_loss": mse_loss.item(),
            "pearson_r": pearson_r.item(),
            "cindex": cindex.item()
        }


    def _move_to_device(self, sample: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Recursively move sample tensors to device"""
        assert isinstance(sample, dict), "Sample must be a dictionary of tensors"
        return {k: v.to(self.device) for k, v in sample.items()}
    

    def _get_cindex(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_mask = targets.reshape((1, -1)) > targets.reshape((-1, 1))
        diff = logits.reshape((1, -1)) - logits.reshape((-1, 1))
        h_one = (diff > 0)
        h_half = (diff == 0)
        cindex = torch.sum(target_mask * h_one + target_mask * h_half * 0.5) / target_mask.sum()
        return cindex
