import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from .preprocess import preprocess_item


@torch.jit.script
def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    """
    Convert multi-dimensional features to single embedding indices.
    Legacy behavior from Dynaformer collator.
    """
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def pad_1d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad 1D tensor with offset +1 (pad id = 0)."""
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad 2D tensor (nodes × features) with offset +1 (pad id = 0)."""
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad position tensor without offset."""
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad attention bias with -inf for masked positions."""
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0  # padding can attend to real nodes
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad edge type tensor (N × N × edge_dim)."""
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pad spatial position matrix with offset +1."""
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x: torch.Tensor, padlen1: int, padlen2: int, padlen3: int) -> torch.Tensor:
    """Pad 3D edge input tensor with offset +1."""
    x = x + 1  # pad id = 0
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def gen_node_type_edge(items: List[Data], max_node_num: int) -> torch.Tensor:
    """
    Generate node type edges encoding ligand-protein interactions.
    
    Node type encodings:
    - ligand-ligand: 3
    - protein-ligand: 4
    - ligand-protein: 5
    - protein-protein: 6
    """
    num_ligand_nodes = [int(item.num_node[0]) for item in items]
    num_protein_nodes = [int(item.num_node[1]) for item in items]
    t = 10  # Number of atom type features
    
    node_type_edges = []
    for idx in range(len(items)):
        n_nodes = num_ligand_nodes[idx] + num_protein_nodes[idx]
        n_ligand_nodes = num_ligand_nodes[idx]
        
        # Initialize node type edge matrix
        node_type_edge = torch.zeros(n_nodes, n_nodes, dtype=torch.long)
        node_type_edge[:n_ligand_nodes, :n_ligand_nodes] = 3  # ligand-ligand
        node_type_edge[n_ligand_nodes:, :n_ligand_nodes] = 4  # protein-ligand
        node_type_edge[:n_ligand_nodes, n_ligand_nodes:] = 5  # ligand-protein
        node_type_edge[n_ligand_nodes:, n_ligand_nodes:] = 6  # protein-protein
        
        # Extract atom types (reverse the convert_to_single_emb offset)
        atom_type = items[idx].x[:, :t] - 1 - torch.arange(0, items[idx].x[:, :t].shape[-1] * 512, 512, dtype=torch.long)
        
        # Create pairwise atom type features
        atom_i = atom_type.unsqueeze(1).repeat(1, n_nodes, 1)
        atom_j = atom_type.unsqueeze(0).repeat(n_nodes, 1, 1)
        atom_type_pair = torch.cat([atom_i, atom_j], dim=-1)
        
        # Combine node type, atom types, and angle information
        node_atom_edge = torch.cat([
            node_type_edge.unsqueeze(2),
            atom_type_pair,
            torch.clamp(torch.div(items[idx].angle.sum(dim=2).to(torch.long), 10, rounding_mode='floor'), 0, 1024).unsqueeze(2),
        ], dim=-1)
        
        # Convert to single embedding indices
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_atom_edge = pad_edge_type_unsqueeze(node_atom_edge, max_node_num)
        node_type_edges.append(node_atom_edge.long())
    
    return torch.cat(node_type_edges)


class GraphormerCollator:
    """
    Modern PyTorch collator matching legacy Dynaformer behavior exactly.
    """
    
    def __init__(
        self,
        max_nodes: int = 512,
        multi_hop_max_dist: int = 20,
        spatial_pos_max: int = 20
    ):
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
    
    def __call__(self, items: List[Data]) -> Dict[str, Any]:
        """
        Collate list of PyG graphs into batched tensors.
        Returns dictionary matching legacy collator output.
        """
        if not items:
            raise ValueError("Empty sample list")
        
        # Filter samples exceeding max_nodes
        original_len = len(items)
        processed_items = []
        for i, item in enumerate(items):
            if item is not None and item.x.size(0) <= self.max_nodes:
                item.idx = i
                item.y = item.y.reshape(-1)
                processed_items.append(preprocess_item(item))
        filtered_len = len(processed_items)
        
        assert filtered_len == original_len, \
            f"Some samples exceed max_nodes: filtered_len={filtered_len}, original_len={original_len}"
        
        # Compute max nodes in this batch
        max_node_num = max(i.x.size(0) for i in processed_items)
        
        # Extract num_node information
        # num_node = torch.stack([item.num_node for item in items])
        
        # Generate node type edges
        node_type_edge = gen_node_type_edge(processed_items, max_node_num)
        
        # Extract all fields from items
        items_tuple = [
            (
                item.idx,
                item.attn_bias,
                item.attn_edge_type,
                item.spatial_pos,
                item.in_degree,
                item.out_degree,
                item.x,
                item.edge_input[:, :, :self.multi_hop_max_dist, :],
                item.y,
                item.pos,
                item.pdbid,
                # item.frame if hasattr(item, "frame") else 0,
                # item.rmsd_lig if hasattr(item, "rmsd_lig") else 0.0,
                # item.rmsd_pro if hasattr(item, "rmsd_pro") else 0.0,
                torch.cat([item.rfscore, item.gbscore, item.ecif]).float() / 100  # 2040-dim
            )
            for item in processed_items
        ]
        
        (
            idxs,
            attn_biases,
            attn_edge_types,
            spatial_poses,
            in_degrees,
            out_degrees,
            xs,
            edge_inputs,
            ys,
            poses,
            pdbids,
            # frames,
            # rmsd_ligs,
            # rmsd_pros,
            fps
        ) = zip(*items_tuple)
        
        # Apply spatial position masking to attention bias
        for idx in range(len(attn_biases)):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= self.spatial_pos_max] = float("-inf")
        
        # Batch all tensors with appropriate padding
        y = torch.cat(ys)
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])  # [B, N, F]
        pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
        edge_input = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, self.multi_hop_max_dist) for i in edge_inputs]
        )
        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        )
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
        
        # frame = torch.LongTensor(frames)
        # rmsd_lig = torch.tensor(rmsd_ligs, dtype=torch.float)
        # rmsd_pro = torch.tensor(rmsd_pros, dtype=torch.float)
        fps = torch.stack(fps).float()
        
        # Return dict matching legacy collator format
        return dict(
            # idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,  # Shape: [batch_size, max_nodes, feature_dim]
            edge_input=edge_input,
            y=y,
            pos=pos,
            pdbid=pdbids,
            # frame=frame,
            # num_node=num_node,
            node_type_edge=node_type_edge,
            # rmsd_lig=rmsd_lig,
            # rmsd_pro=rmsd_pro,
            fp=fps
        )