from pathlib import Path
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
class AMPConfig:
    """Configuration for Automatic Mixed Precision (AMP)"""
    use_amp: bool = True
    scaler_init_scale: float = 16384.0
    scaler_growth_factor: float = 2.0
    scaler_backoff_factor: float = 0.5
    scaler_growth_interval: int = 2000

@dataclass
class FlagConfig:
    """Configuration for FLAG"""
    flag: bool = True
    flag_m: int = 3
    flag_step_size: float = 1e-3
    flag_mag: float = 1e-3

@dataclass
class TrainerConfig:
    """Configuration for training loop"""
    flag_config: FlagConfig
    amp_config: AMPConfig
    clip_norm: Optional[float] = None
    save_interval: int = 10
    num_epochs: int = 100
    checkpoint_dir: Path = Path("checkpoints")
    start_epoch: int = 1


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: Union[str, nn.Module],
        config: TrainerConfig,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler]=None,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.criterion = get_criterion(criterion)() if isinstance(criterion, str) else criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.scaler = torch.GradScaler(
            device,
            init_scale=config.amp_config.scaler_init_scale,
            growth_factor=config.amp_config.scaler_growth_factor,
            backoff_factor=config.amp_config.scaler_backoff_factor,
            growth_interval=config.amp_config.scaler_growth_interval
        ) if config.amp_config.use_amp else None
        self.update_num = 0
        
        self.config.clip_norm = config.clip_norm if config.clip_norm and config.clip_norm > 0 else None
        self.train_step = self._train_step_with_flag if config.flag_config.flag else self._train_step_standard


    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
    ):
        """Main training loop"""
        logging.info(f"Starting training, FLAG: {self.config.flag_config.flag}, AMP: {self.config.amp_config.use_amp}")
        # use current datetime as suffix for checkpoint directory
        for epoch in range(self.config.start_epoch, self.config.num_epochs + 1):
            logging.info(f"Starting epoch {epoch}/{self.config.num_epochs}, lr: {self.optimizer.param_groups[0]['lr']:.4e}")

            train_stats = self.train_epoch(train_dataloader, epoch)
            logging.info(f"Epoch {epoch} training loss: {train_stats['loss']:.4f}")
            
            val_stats = self.validate(val_dataloader)
            logging.info(f"Epoch {epoch} validation loss: {val_stats['loss']:.4f}")

            if self.scheduler is not None and train_stats['step_successful']:
                self.scheduler.step()

            if epoch % self.config.save_interval == 0:
                checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'epoch': epoch,
                    'train_stats': train_stats,
                    'val_stats': val_stats,
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
            loss, sample_size, logging_output, step_successful = self.train_step(sample)
            
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
            "num_samples": num_samples,
            "step_successful": step_successful
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
            # print(type(sample['y']), sample['y'].shape, sample['y'].dtype, sample['y'].device)

            with torch.autocast(self.device, enabled=self.config.amp_config.use_amp):
                criterion_output = self.criterion(self.model, sample)
                loss = criterion_output.loss # shape: (), scalar tensor
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
        # ignore_grad: bool=False
    ) -> Tuple[torch.Tensor, int, Dict[str, float], bool]:
        """Standard training step without FLAG"""
        self.update_num += 1
        
        # Move sample to device
        sample = self._move_to_device(sample)
        
        self.optimizer.zero_grad()
        loss, sample_size, logging_output = self._forward_and_backward(sample)
        step_successful = self._optimizer_step()

        return loss.detach(), sample_size, logging_output, step_successful


    def _train_step_with_flag(
        self,
        sample: Dict[str, torch.Tensor],
        # ignore_grad: bool=False
    ) -> Tuple[torch.Tensor, int, Dict[str, float], bool]:
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
        total_loss = 0.0

        # Iterative perturbation optimization
        for i in range(self.config.flag_config.flag_m):
            loss, sample_size, logging_output = self._forward_and_backward(sample, loss_divisive_scale=self.config.flag_config.flag_m)
            total_loss += loss.detach().item()
            
            # Update perturbation
            if i < self.config.flag_config.flag_m - 1:
                perturb_data = self._update_perturbation(perturb)
                perturb.data = perturb_data
                perturb.grad.zero_()
                sample["perturb"] = perturb
        
        total_loss += loss.detach()
        
        # Optimizer step
        step_successful = self._optimizer_step()
        
        logging_output["loss"] = total_loss.item()
        return total_loss, sample_size, logging_output, step_successful
    

    def _forward_and_backward(
            self,
            sample: Dict[str, torch.Tensor],
            loss_divisive_scale: float = 1.0
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """Helper function to perform forward and backward pass"""
        with torch.autocast(self.device, enabled=self.config.amp_config.use_amp):
            criterion_output = self.criterion(self.model, sample)
            loss = criterion_output.loss # shape: (), scalar tensor
            sample_size = criterion_output.sample_size
            logging_output = criterion_output.logging_output

        loss /= loss_divisive_scale
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss, sample_size, logging_output
    

    def _optimizer_step(self) -> bool:
        """Helper function to perform optimizer step with optional grad scaling"""
        step_successful = True
        if self.scaler is not None:
            if self.config.clip_norm:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
            self.scaler.step(self.optimizer)
            scale_before = self.scaler.get_scale()
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            if scale_after <= scale_before * self.config.amp_config.scaler_backoff_factor:
                step_successful = False
                logging.info(f"Update num {self.update_num}: Grad scaler step, scale {scale_before} -> {scale_after}, step failed")
        else:
            if self.config.clip_norm:
                # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
            self.optimizer.step()
        
        return step_successful


    def _initialize_perturbation(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> torch.Tensor:
        """Initialize perturbation tensor for FLAG"""
        if self.config.flag_config.flag_mag > 0:
            perturb = torch.empty(*shape, device=device).uniform_(-1, 1)
            perturb *= self.config.flag_config.flag_mag / math.sqrt(shape[-1])
        else:
            perturb = torch.empty(*shape, device=device).uniform_(
                -self.config.flag_config.flag_step_size, self.config.flag_config.flag_step_size
            )
        perturb.requires_grad_(True)
        return perturb


    def _update_perturbation(self, perturb: torch.Tensor) -> torch.Tensor:
        """Update perturbation using gradient ascent"""
        perturb_data = perturb.detach() + self.config.flag_config.flag_step_size * torch.sign(
            perturb.grad.detach() # grad exists here because we set requires_grad=True for perturb
        )
        
        # Project perturbation to magnitude bound
        if self.config.flag_config.flag_mag > 0:
            perturb_data_norm = torch.norm(perturb_data, dim=-1, keepdim=False)
            exceed_mask = (perturb_data_norm > self.config.flag_config.flag_mag).to(perturb_data.dtype)
            reweights = (
                self.config.flag_config.flag_mag / (perturb_data_norm + 1e-8) * exceed_mask + (1 - exceed_mask)
            ).unsqueeze(-1)
            perturb_data *= reweights
        
        return perturb_data.detach()


    def _move_to_device(self, sample: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Recursively move sample tensors to device"""
        assert isinstance(sample, dict), "Sample must be a dictionary of tensors"
        return {k: v.to(self.device) for k, v in sample.items()}
