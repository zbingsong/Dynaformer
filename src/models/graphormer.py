# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union

from src.modules import init_graphormer_params, GraphormerGraphEncoder

logger = logging.getLogger(__name__)


@dataclass
class GraphormerConfig:
    """Configuration class for Graphormer model."""
    # Basic model architecture
    encoder_embed_dim: int = 768
    encoder_layers: int = 12
    encoder_attention_heads: int = 32
    encoder_ffn_embed_dim: int = 768
    num_classes: int = 1
    max_nodes: int = 512
    
    # Dropout settings
    dropout: float = 0.1
    attention_dropout: float = 0.1
    act_dropout: float = 0.1
    layerdrop: float = 0.1

    # Model behavior
    encoder_normalize_before: bool = True
    apply_graphormer_init: bool = False
    activation_fn: str = "gelu"
    embed_scale: Optional[float] = None
    sandwich_ln: bool = False
    
    # Graph-specific parameters
    num_atoms: int = 512*9
    num_in_degree: int = 512
    num_out_degree: int = 512
    num_edges: int = 512*3
    num_spatial: int = 512
    num_edge_dis: int = 128
    edge_type: str = "multi_hop"
    multi_hop_max_dist: int = 5

    # 3D encoder settings
    dist_head: str = "none"
    num_dist_head_kernel: int = 128
    num_edge_types: int = 512*16

    # Additional features
    fingerprint: bool = True
    sample_weight_estimator: bool = False
    sample_weight_estimator_pat: str = "pdbbind"


class GraphormerModel(nn.Module):
    """Modern PyTorch implementation of Graphormer model without Fairseq dependencies."""
    
    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.config = config
        
        # Create encoder
        self.encoder = GraphormerEncoder(config)
        
        if config.apply_graphormer_init:
            self.apply(init_graphormer_params)

    @classmethod
    def from_args(cls, args) -> 'GraphormerModel':
        """Create model from legacy args object for compatibility."""
        config = GraphormerConfig()
        
        # Map args to config
        for field_name in config.__dataclass_fields__:
            if hasattr(args, field_name):
                setattr(config, field_name, getattr(args, field_name))
        
        # Handle special cases
        if hasattr(args, 'tokens_per_sample') and not hasattr(args, 'max_nodes'):
            config.max_nodes = args.tokens_per_sample
        
        if hasattr(args, 'embed_scale') and args.embed_scale > 0:
            config.embed_scale = args.embed_scale
            
        logger.info(f"Created GraphormerModel with config: {config}")
        return cls(config)

    def max_nodes(self) -> int:
        return self.encoder.max_nodes

    def forward(
            self, 
            batched_data: dict[str, torch.Tensor],
            perturb: Optional[torch.Tensor]=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.encoder(
            batched_data, 
            perturb=perturb,
        )


class GraphormerEncoder(nn.Module):
    """Modern PyTorch encoder implementation without Fairseq dependencies."""
    
    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.config = config
        self.max_nodes = config.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            # Graph-specific parameters
            num_atoms=config.num_atoms,
            num_in_degree=config.num_in_degree,
            num_out_degree=config.num_out_degree,
            num_edges=config.num_edges,
            num_spatial=config.num_spatial,
            num_edge_dis=config.num_edge_dis,
            edge_type=config.edge_type,
            multi_hop_max_dist=config.multi_hop_max_dist,
            # Architecture parameters
            num_encoder_layers=config.encoder_layers,
            embedding_dim=config.encoder_embed_dim,
            ffn_embedding_dim=config.encoder_ffn_embed_dim,
            num_attention_heads=config.encoder_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.act_dropout,
            layerdrop=config.layerdrop,
            encoder_normalize_before=config.encoder_normalize_before,
            apply_graphormer_init=config.apply_graphormer_init,
            activation_fn=config.activation_fn,
            embed_scale=config.embed_scale,
            dist_head=config.dist_head,
            sandwich_ln=config.sandwich_ln,
            # 3D encoder parameters
            num_dist_head_kernel=config.num_dist_head_kernel,
            num_edge_types=config.num_edge_types,
        )

        # Output layers
        self.masked_lm_pooler = nn.Linear(
            config.encoder_embed_dim, config.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            config.encoder_embed_dim, config.encoder_embed_dim
        )
        
        # Activation function
        if config.activation_fn == "relu":
            self.activation_fn = F.relu
        elif config.activation_fn == "gelu":
            self.activation_fn = F.gelu
        else:
            raise NotImplementedError(f"Unknown activation function: {config.activation_fn}")
            
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)

        # Output head
        self.embed_out = nn.Linear(
            config.encoder_embed_dim, config.num_classes
        )
            
        # Sample weight estimator
        self.sample_weight_estimator = config.sample_weight_estimator
        self.sample_weight_estimator_pat = config.sample_weight_estimator_pat
        
        # Fingerprint network
        if config.fingerprint:
            self.fpnn = nn.Sequential(
                nn.Linear(2040, config.encoder_embed_dim),
                nn.GELU(),
                nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim)
            )
            self.reducer = nn.Linear(config.encoder_embed_dim * 2, config.encoder_embed_dim)
        else:
            self.fpnn = None

    def reset_output_layer_parameters(self) -> None:
        """Reset output layer parameters."""
        self.embed_out.reset_parameters()

    def forward(
            self, 
            batched_data: dict[str, torch.Tensor], 
            perturb: Optional[torch.Tensor]=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the encoder."""
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )
        
        # inner_states: List[Tensor] of shape [seq_len, batch, hidden]
        # Take the last layer and transpose to [batch, seq_len, hidden]
        x = inner_states[-1].transpose(0, 1)
        
        # Apply transformation and layer norm
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        
        # x: [batch, seq_len, hidden]
        if self.fpnn is not None:
            # Use fingerprint features
            fpemb = self.fpnn(batched_data["fp"])  # [batch, hidden]
            # Combine graph token (first position) with fingerprint
            x = self.reducer(torch.cat([x[:, 0, :], fpemb], dim=1))
        else:
            # Use only graph token (first position)
            x = x[:, 0, :]  # [batch, hidden]
        
        # Project to output classes
        x = self.embed_out(x) # shape [batch, num_classes]

        # Sample weight estimation
        # if self.sample_weight_estimator:
        #     weight = torch.ones(x.shape, dtype=x.dtype, device=x.device) * 0.01
        #     wmask = torch.ones(weight.shape, dtype=torch.bool, device=weight.device)
        #     wones = torch.ones(weight.shape, dtype=weight.dtype, device=weight.device)
            
        #     for idx, pdbid in enumerate(batched_data['pdbid']):
        #         if pdbid.endswith(self.sample_weight_estimator_pat):
        #             wmask[idx] = False
                    
        #     weight = torch.where(wmask, weight, wones)
        #     return x, weight
            
        return x


# Configuration builders for different model sizes

def get_base_config() -> GraphormerConfig:
    """Get base Graphormer configuration."""
    return GraphormerConfig(
        dropout=0.1,
        attention_dropout=0.1,
        act_dropout=0.0,
        encoder_ffn_embed_dim=4096,
        encoder_layers=6,
        encoder_attention_heads=8,
        encoder_embed_dim=1024,
        share_encoder_input_output_embed=False,
        no_token_positional_embeddings=False,
        apply_graphormer_init=False,
        activation_fn="gelu",
        encoder_normalize_before=True,
    )


def get_graphormer_base_config() -> GraphormerConfig:
    """Get Graphormer base configuration."""
    config = get_base_config()
    config.encoder_embed_dim = 768
    config.encoder_layers = 12
    config.encoder_attention_heads = 32
    config.encoder_ffn_embed_dim = 768
    config.activation_fn = "gelu"
    config.encoder_normalize_before = True
    config.apply_graphormer_init = False
    # config.share_encoder_input_output_embed = False
    # config.no_token_positional_embeddings = False
    return config


def get_graphormer_slim_config() -> GraphormerConfig:
    """Get Graphormer slim configuration."""
    config = get_base_config()
    config.encoder_embed_dim = 80
    config.encoder_layers = 12
    config.encoder_attention_heads = 8
    config.encoder_ffn_embed_dim = 80
    config.activation_fn = "gelu"
    config.encoder_normalize_before = True
    config.apply_graphormer_init = False
    # config.share_encoder_input_output_embed = False
    # config.no_token_positional_embeddings = False
    return config


def get_graphormer_large_config() -> GraphormerConfig:
    """Get Graphormer large configuration."""
    config = get_base_config()
    config.encoder_embed_dim = 1024
    config.encoder_layers = 24
    config.encoder_attention_heads = 32
    config.encoder_ffn_embed_dim = 1024
    config.activation_fn = "gelu"
    config.encoder_normalize_before = True
    config.apply_graphormer_init = False
    # config.share_encoder_input_output_embed = False
    # config.no_token_positional_embeddings = False
    return config
