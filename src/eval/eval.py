import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Union
from src.criterions import get_criterion


class Evaluator:
    def __init__(
            self, 
            model: nn.Module, 
            criterion: Union[str, nn.Module], 
            device: str="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.criterion = get_criterion(criterion)() if isinstance(criterion, str) else criterion
        self.device = device


    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for sample in dataloader:
                sample = self._move_to_device(sample)
                criterion_output = self.criterion(self.model, sample)
                loss = criterion_output.loss
                sample_size = criterion_output.sample_size

                total_loss += loss.item()
                total_samples += sample_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss


    def _move_to_device(self, sample: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Recursively move sample tensors to device"""
        assert isinstance(sample, dict), "Sample must be a dictionary of tensors"
        return {k: v.to(self.device) for k, v in sample.items()}