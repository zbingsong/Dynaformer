import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree


def pad_1d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """
    Pad 1D tensor and add batch dimension.
    Equivalent to legacy collator padding logic.
    """
    if x.size(0) > padlen:
        raise ValueError(f"Cannot pad tensor of size {x.size(0)} to {padlen}")
    
    if x.dim() == 1:
        padded = F.pad(x, (0, padlen - x.size(0)))
        return padded.unsqueeze(0)  # Add batch dim
    elif x.dim() == 2:
        padded = F.pad(x, (0, 0, 0, padlen - x.size(0)))
        return padded.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected tensor dimension: {x.dim()}")


def pad_2d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad 2D tensor to square matrix."""
    if x.size(0) > padlen or x.size(1) > padlen:
        raise ValueError(f"Cannot pad tensor of size {x.shape} to ({padlen}, {padlen})")
    
    padded = F.pad(x, (0, padlen - x.size(1), 0, padlen - x.size(0)))
    return padded.unsqueeze(0)


def pad_edge_type_unsqueeze(edge_attr: torch.Tensor, edge_index: torch.Tensor, 
                           max_nodes: int) -> torch.Tensor:
    """
    Convert edge attributes to padded adjacency matrix.
    Replaces legacy Cython algos.pyx functionality.
    """
    n_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0
    
    # Create edge type matrix
    edge_type_matrix = torch.zeros(max_nodes, max_nodes, dtype=torch.long)
    
    if edge_index.numel() > 0:
        row, col = edge_index[0], edge_index[1]
        
        # Use first edge attribute dimension as edge type
        if edge_attr.numel() > 0:
            edge_types = edge_attr[:, 0].long() if edge_attr.dim() > 1 else edge_attr.long()
        else:
            edge_types = torch.ones(row.size(0), dtype=torch.long)
        
        # Fill adjacency matrix
        edge_type_matrix[row, col] = edge_types
    
    return edge_type_matrix.unsqueeze(0)


def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    """
    Convert multi-dimensional node features to single embedding indices.
    Replaces legacy Graphormer node feature handling.
    """
    feature_dims = x.size(-1)
    if feature_dims == 1:
        return x.long()
    
    # Convert each feature dimension to unique indices
    multipliers = torch.arange(feature_dims, device=x.device) * offset
    return (x * multipliers[None, :]).sum(dim=-1).long()


class GraphormerCollator:
    """
    Modern PyTorch collator for Dynaformer graphs.
    Replaces legacy Fairseq collator with pure PyTorch implementation.
    """
    
    def __init__(
        self,
        max_nodes: int = 600,
        multi_hop_max_dist: int = 20,
        spatial_pos_max: int = 20
    ):
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
    
    def __call__(self, samples: List[Data]) -> Dict[str, torch.Tensor]:
        """
        Collate list of PyG graphs into batched tensors.
        
        Args:
            samples: List of PyG Data objects
            
        Returns:
            Dictionary with batched tensors for model input
        """
        if not samples:
            raise ValueError("Empty sample list")
        
        batch_size = len(samples)
        
        # Initialize output tensors
        batched_data = {
            'x': torch.zeros(batch_size, self.max_nodes, dtype=torch.long),
            'edge_input': torch.zeros(batch_size, self.max_nodes, self.max_nodes, dtype=torch.long),
            'attn_bias': torch.zeros(batch_size, self.max_nodes + 1, self.max_nodes + 1),
            'spatial_pos': torch.zeros(batch_size, self.max_nodes, self.max_nodes, dtype=torch.long),
            'in_degree': torch.zeros(batch_size, self.max_nodes, dtype=torch.long),
            'out_degree': torch.zeros(batch_size, self.max_nodes, dtype=torch.long),
            'fingerprint': torch.zeros(batch_size, 2040),  # RF+GB+ECIF
            'targets': torch.zeros(batch_size)
        }
        
        # Process each sample
        for i, sample in enumerate(samples):
            try:
                self._process_sample(sample, batched_data, i)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Skip problematic samples
                continue
        
        return batched_data
    
    def _process_sample(self, sample: Data, batched_data: Dict[str, torch.Tensor], idx: int):
        """Process individual sample into batch."""
        n_nodes = sample.x.size(0)
        
        if n_nodes > self.max_nodes:
            raise ValueError(f"Sample has {n_nodes} nodes > max_nodes {self.max_nodes}")
        
        # Node features
        if sample.x.size(-1) > 1:
            # Convert multi-dim features to single embedding indices
            node_features = convert_to_single_emb(sample.x)
        else:
            node_features = sample.x.squeeze(-1).long()
        
        batched_data['x'][idx, :n_nodes] = node_features
        
        # Edge features and adjacency
        if hasattr(sample, 'edge_index') and sample.edge_index.numel() > 0:
            edge_attr = getattr(sample, 'edge_attr', torch.ones(sample.edge_index.size(1), 1))
            edge_matrix = pad_edge_type_unsqueeze(edge_attr, sample.edge_index, self.max_nodes)
            batched_data['edge_input'][idx] = edge_matrix.squeeze(0)
            
            # Compute degrees
            row, col = sample.edge_index
            in_deg = degree(col, num_nodes=n_nodes, dtype=torch.long)
            out_deg = degree(row, num_nodes=n_nodes, dtype=torch.long)
            batched_data['in_degree'][idx, :n_nodes] = in_deg
            batched_data['out_degree'][idx, :n_nodes] = out_deg
        
        # Spatial positions (from 3D coordinates)
        if hasattr(sample, 'pos') and sample.pos.numel() > 0:
            spatial_pos = self._compute_spatial_pos(sample.pos)
            batched_data['spatial_pos'][idx, :n_nodes, :n_nodes] = spatial_pos
        
        # Attention bias (structural bias)
        attn_bias = self._compute_attention_bias(sample, n_nodes)
        batched_data['attn_bias'][idx, :n_nodes+1, :n_nodes+1] = attn_bias
        
        # Molecular fingerprint
        if hasattr(sample, 'fingerprint'):
            batched_data['fingerprint'][idx] = sample.fingerprint.float()
        
        # Target
        if hasattr(sample, 'y'):
            batched_data['targets'][idx] = sample.y.item() if sample.y.numel() == 1 else sample.y[0]
    
    def _compute_spatial_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial position matrix from 3D coordinates.
        Based on Graphormer 3D distance encoding.
        """
        n_nodes = pos.size(0)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(pos, pos, p=2)
        
        # Discretize distances
        spatial_pos = torch.clamp(
            (dist_matrix * 2).long(),  # Scale factor for discretization
            0, self.spatial_pos_max - 1
        )
        
        return spatial_pos
    
    def _compute_attention_bias(self, sample: Data, n_nodes: int) -> torch.Tensor:
        """
        Compute attention bias matrix.
        Includes graph token (virtual node) connection.
        """
        bias = torch.zeros(n_nodes + 1, n_nodes + 1)
        
        # Connect all nodes to graph token (index 0)
        bias[0, 1:n_nodes+1] = 0.0  # Graph token to nodes
        bias[1:n_nodes+1, 0] = 0.0  # Nodes to graph token
        
        # Node-to-node connections based on edges
        if hasattr(sample, 'edge_index') and sample.edge_index.numel() > 0:
            row, col = sample.edge_index
            # Add 1 to account for graph token offset
            bias[row + 1, col + 1] = 0.0
        
        return bias
