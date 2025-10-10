import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from typing import Tuple


@torch.jit.script
def convert_to_single_emb(x: torch.Tensor, offset: int = 512):
    """Convert multi-dimensional features to single embedding space."""
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def floyd_warshall(adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shortest paths between all pairs of nodes using Floyd-Warshall algorithm.
    
    Args:
        adjacency_matrix: numpy array of shape [n, n] representing graph adjacency
        
    Returns:
        M: shortest path distance matrix
        path: intermediate nodes for path reconstruction
    """
    n = adjacency_matrix.shape[0]
    M = adjacency_matrix.astype(np.int64, copy=True)
    path = -1 * np.ones([n, n], dtype=np.int64)
    
    # Set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i][k] + M[k][j]
                if M[i][j] > cost_ikkj:
                    M[i][j] = cost_ikkj
                    path[i][j] = k
    
    # Set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510
    
    return M, path


def get_all_edges(path: np.ndarray, i: int, j: int) -> list[int]:
    """Reconstruct path from i to j using intermediate nodes."""
    k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist: int, path: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
    """
    Generate edge input features for all shortest paths.
    
    Args:
        max_dist: maximum distance in the graph
        path: path matrix from floyd_warshall
        edge_feat: edge features of shape [n, n, feat_dim]
        
    Returns:
        edge_fea_all: edge features along shortest paths [n, n, max_dist, feat_dim]
    """
    n = path.shape[0]
    edge_fea_all = -1 * np.ones([n, n, max_dist, edge_feat.shape[-1]], dtype=np.int64)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path[i][j] == 510:
                continue
            path_ij = [i] + get_all_edges(path, i, j) + [j]
            num_path = len(path_ij) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat[path_ij[k], path_ij[k+1], :]
    
    return edge_fea_all


def gen_angle_dist(item: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate angle and distance features for graph.
    
    Args:
        item: PyG data object with edge_index and pos attributes
        
    Returns:
        angle: angle features [N, N, n_angle]
        dists: distance features [N, N, n_angle]
    """
    edge_index: torch.Tensor = item.edge_index if hasattr(item, 'edge_index') else item['edge_index']
    pos: torch.Tensor = item.pos if hasattr(item, 'pos') else item['pos']
    
    n_node = pos.shape[0]
    dense_adj = to_dense_adj(edge_index, max_num_nodes=n_node).squeeze()
    n_angle = 28
    
    neighbors = torch.zeros(n_node, n_angle, dtype=torch.long) - 1
    for i in range(n_node):
        n = dense_adj[i].nonzero().squeeze(dim=1)[:n_angle]
        neighbors[i, :n.shape[0]] = n
    
    # Compute edge vectors i->j
    ijs = (pos[edge_index[1]] - pos[edge_index[0]])  # n_edge x 3
    nijs = ijs.norm(dim=-1, keepdim=True)
    ijs /= nijs
    
    # Compute edge vectors i->k for neighbors
    iks = pos[neighbors[edge_index[0]]] - (pos[edge_index[0]]).unsqueeze(1)  # n_edge x n_angle x 3
    niks = iks.norm(dim=-1, keepdim=True)
    iks /= niks
    mask = neighbors[edge_index[0]] < 0
    
    # Compute angles
    cos = torch.bmm(iks, ijs.unsqueeze(2)).squeeze()
    out = torch.arccos(cos) * 180 / np.pi
    out[mask] = -1
    out.nan_to_num_(1e-6)
    out = out + 1  # angle = 0 is padding idx

    angle = to_dense_adj(edge_index, edge_attr=out).squeeze()
    dists = to_dense_adj(edge_index, edge_attr=niks.squeeze()).squeeze()

    return angle, dists


def preprocess_item(item: Data) -> Data:
    """
    Preprocess a PyG graph data item by adding structural encodings.
    
    Args:
        item: PyG data object with at minimum edge_attr, edge_index, and x attributes
        
    Returns:
        item: The same item with additional attributes:
            - x: converted node features
            - attn_bias: attention bias matrix
            - attn_edge_type: edge type encodings
            - spatial_pos: shortest path distances
            - in_degree: node in-degrees
            - out_degree: node out-degrees
            - edge_input: edge features along shortest paths
            - angle: angle features (if pos available)
            - dists: distance features (if pos available)
    """
    edge_attr: torch.Tensor = item.edge_attr
    edge_index: torch.Tensor = item.edge_index
    x: torch.Tensor = item.x
    
    N = x.size(0)
    x = convert_to_single_emb(x)
    
    # Node adjacency matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    
    # Edge feature encoding
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    
    # Compute shortest paths
    shortest_path_result, path = floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    
    # Compute angle and distance features if position available
    if hasattr(item, 'pos') and item.pos is not None:
        angle, dists = gen_angle_dist(item)
        item.angle = angle.to(torch.float)
        item.dists = dists.to(torch.float)
    
    # Combine all features
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()
    
    return item
