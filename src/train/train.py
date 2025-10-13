from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import logging
from typing import Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass

from src.criterions import get_criterion


@dataclass
class TrainingConfig:
    """Configuration for training loop"""
    use_flag: bool = False
    flag_m: int = 3
    flag_step_size: float = 1e-3
    flag_mag: float = 1e-3
    use_amp: bool = True
    log_interval: int = 10
    num_epochs: int = 100
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: Union[str, nn.Module],
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.criterion = get_criterion(criterion)() if isinstance(criterion, str) else criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scaler = torch.GradScaler(device) if config.use_amp else None
        self.update_num = 0

        self.train_step = self._train_step_with_flag if config.use_flag else self._train_step_standard


    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
    ):
        """Main training loop"""
        # use current datetime as suffix for checkpoint directory
        for epoch in range(1, self.config.num_epochs + 1):
            train_stats = self.train_epoch(train_dataloader, epoch)
            logging.info(f"Epoch {epoch} training loss: {train_stats['loss']:.4f}")
            
            val_stats = self.validate(val_dataloader)
            logging.info(f"Epoch {epoch} validation loss: {val_stats['loss']:.4f}")

            if epoch % self.config.log_interval == 0:
                checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'config': self.config
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")


    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for batch_idx, sample in enumerate(dataloader):
            loss, sample_size, logging_output = self.train_step(sample)
            
            total_loss += loss.item() # set reduction="none" in criterion already
            num_samples += sample_size

            # if (batch_idx+1) % self.config.log_interval == 0:
            #     avg_loss = total_loss / num_samples if num_samples > 0 else 0
            #     logger.info(
            #         f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
            #         f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}"
            #     )
        
        return {
            "loss": total_loss / num_samples if num_samples > 0 else 0,
            "num_samples": num_samples
        }


    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        for sample in dataloader:
            sample = self._move_to_device(sample)
            
            with torch.autocast(self.device, enabled=self.config.use_amp):
                criterion_output = self.criterion(self.model, sample)
                loss = criterion_output.loss
                sample_size = criterion_output.sample_size
                # logging_output = criterion_output.logging_output

            total_loss += loss.item()
            num_samples += sample_size
        
        return {
            "loss": total_loss / num_samples if num_samples > 0 else 0,
            "num_samples": num_samples
        }


    def _train_step_standard(
        self,
        sample: Dict[str, torch.Tensor],
        ignore_grad: bool=False
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """Standard training step without FLAG"""
        self.update_num += 1
        
        # Move sample to device
        sample = self._move_to_device(sample)
        
        self.optimizer.zero_grad()
        
        with torch.autocast(self.device, enabled=self.config.use_amp):
            criterion_output = self.criterion(self.model, sample)
            loss = criterion_output.loss
            sample_size = criterion_output.sample_size
            logging_output = criterion_output.logging_output
            loss *= (ignore_grad == False)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss.detach(), sample_size, logging_output


    def _train_step_with_flag(
        self,
        sample: Dict[str, torch.Tensor],
        ignore_grad: bool=False
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """
        Training step with FLAG (Free Large-scale Adversarial Augmentation on Graphs)
        
        Implements adversarial training by adding learned perturbations to node embeddings.
        """
        self.update_num += 1
        
        # Move sample to device
        sample = self._move_to_device(sample)
        
        # Initialize perturbation
        batched_data = sample["x"] # shape (batch_size, num_batch_max_nodes, embedding_dim)
        n_graph, n_node = batched_data.shape[:2]
        perturb_shape = (n_graph, n_node, self.model.config.encoder_embed_dim)
        
        perturb = self._initialize_perturbation(perturb_shape, batched_data.device)
        sample["perturb"] = perturb
        
        # First forward pass
        self.optimizer.zero_grad()

        with torch.autocast(self.device, enabled=self.config.use_amp):
            criterion_output = self.criterion(self.model, sample)
            loss = criterion_output.loss
            sample_size = criterion_output.sample_size
            logging_output = criterion_output.logging_output
            loss *= (ignore_grad == False)
        
        loss /= self.config.flag_m
        total_loss = torch.zeros_like(loss)
        
        # Iterative perturbation optimization
        for _ in range(self.config.flag_m - 1):
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.detach()
            
            # Update perturbation
            perturb_data = self._update_perturbation(perturb)
            perturb.data = perturb_data
            perturb.grad.zero_()
            
            sample["perturb"] = perturb
            
            # Forward pass with updated perturbation
            with torch.autocast(self.device, enabled=self.config.use_amp):
                criterion_output = self.criterion(self.model, sample)
                loss = criterion_output.loss
                sample_size = criterion_output.sample_size
                logging_output = criterion_output.logging_output
                if ignore_grad:
                    loss = loss * 0
            
            loss = loss / self.config.flag_m

        # Final backward pass for perturbation, does not update perturbation
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.detach()
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        logging_output["loss"] = total_loss.item()
        return total_loss, sample_size, logging_output


    def _initialize_perturbation(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> torch.Tensor:
        """Initialize perturbation tensor for FLAG"""
        if self.config.flag_mag > 0:
            perturb = torch.empty(*shape, device=device).uniform_(-1, 1)
            perturb *= self.config.flag_mag / math.sqrt(shape[-1])
        else:
            perturb = torch.empty(*shape, device=device).uniform_(
                -self.config.flag_step_size, self.config.flag_step_size
            )
        perturb.requires_grad_(True)
        return perturb


    def _update_perturbation(self, perturb: torch.Tensor) -> torch.Tensor:
        """Update perturbation using gradient ascent"""
        perturb_data = perturb.detach() + self.config.flag_step_size * torch.sign(
            perturb.grad.detach() # grad exists here because we set requires_grad=True for perturb
        )
        
        # Project perturbation to magnitude bound
        if self.config.flag_mag > 0:
            perturb_data_norm = torch.norm(perturb_data, dim=-1, keepdim=False)
            exceed_mask = (perturb_data_norm > self.config.flag_mag).to(perturb_data.dtype)
            reweights = (
                self.config.flag_mag / (perturb_data_norm + 1e-8) * exceed_mask + (1 - exceed_mask)
            ).unsqueeze(-1)
            perturb_data *= reweights
        
        return perturb_data.detach()


    def _move_to_device(self, sample: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Recursively move sample tensors to device"""
        assert isinstance(sample, dict), "Sample must be a dictionary of tensors"
        return {k: v.to(self.device) for k, v in sample.items()}
