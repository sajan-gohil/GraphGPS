"""
Spectral Attention Layer using Chebyshev polynomials.
Based on the paper: Spectrally-Aligned Attention Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    get_laplacian, add_self_loops, to_dense_batch
)


class ChebyshevFilter(nn.Module):
    """
    Applies a learnable spectral filter using Chebyshev polynomials.
    Output = Sum_k (theta_k * T_k(L) * X)
    
    The filter applies Chebyshev polynomials of the Laplacian matrix
    to learn band-pass filters in the spectral domain.
    """
    
    def __init__(self, in_channels, K=3):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        
        # Learnable coefficients for the filter (one per order k)
        # Initialize to mimic a low-pass filter (decaying with k)
        self.coeffs = nn.Parameter(torch.randn(K))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.1)

    def forward(self, x, edge_index):
        """
        Apply spectral filter using Chebyshev polynomial expansion.
        
        Args:
            x: [N, dim] Node features
            edge_index: [2, E] Edge indices
            
        Returns:
            [N, dim] Filtered features
        """
        # Ensure self loops are present for stability
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalized Laplacian: L = I - D^-0.5 A D^-0.5
        edge_index_L, edge_weight_L = get_laplacian(
            edge_index, normalization='sym'
        )
        
        # Helper function for sparse matrix-vector multiplication
        def sparse_mm(idx, wt, mat):
            """Sparse matrix multiplication."""
            return torch.sparse.mm(
                torch.sparse_coo_tensor(
                    idx, wt, (mat.size(0), mat.size(0)), device=mat.device
                ),
                mat
            )
        
        # Initialize Chebyshev polynomials: T_0(x) = x, T_1(x) = (L-I)x
        Tx_0 = x
        
        # For Tx_1, we need (L - I)x = Lx - x
        Lx = sparse_mm(edge_index_L, edge_weight_L, x)
        Tx_1 = Lx - x
        
        # Accumulate output
        out = self.coeffs[0] * Tx_0 + self.coeffs[1] * Tx_1
        
        Tx_prev = Tx_1
        Tx_prev2 = Tx_0
        
        # Chebyshev recurrence: T_k(x) = 2 * (L-I) * T_{k-1} - T_{k-2}
        for k in range(2, self.K):
            L_Tx_prev = sparse_mm(edge_index_L, edge_weight_L, Tx_prev)
            term1 = 2 * (L_Tx_prev - Tx_prev)
            Tx_k = term1 - Tx_prev2
            
            out = out + self.coeffs[k] * Tx_k
            
            Tx_prev2 = Tx_prev
            Tx_prev = Tx_k
        
        return out


class SpectralAttentionLayer(nn.Module):
    """
    Spectrally-Decoupled Attention (SDA) Layer.
    
    Uses Chebyshev filters to learn band-pass filters on the graph Laplacian
    for Query and Key projections in multi-head attention.
    """
    
    def __init__(self, embed_dim, num_heads, K=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.head_dim = embed_dim // num_heads
        
        # Linear Projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Spectral Filters for Q and K
        # Each head gets its own learnable band-pass filter
        self.filters_q = nn.ModuleList([
            ChebyshevFilter(self.head_dim, K) for _ in range(num_heads)
        ])
        self.filters_k = nn.ModuleList([
            ChebyshevFilter(self.head_dim, K) for _ in range(num_heads)
        ])
        
        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass of spectral attention layer.
        
        Args:
            x: [N, embed_dim] Node features
            edge_index: [2, E] Edge indices
            batch: [N] Batch indices for graphs in batch
            
        Returns:
            [N, embed_dim] Output features
        """
        # Store input for residual connection
        residual = x
        x = self.norm1(x)
        
        # 1. Linear projections
        q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(x).view(-1, self.num_heads, self.head_dim)
        
        # 2. Apply spectral filters per head
        q_spec_list = []
        k_spec_list = []
        
        for h in range(self.num_heads):
            # Extract head-specific features: [N, head_dim]
            q_h = q[:, h, :]
            k_h = k[:, h, :]
            
            # Apply band-pass filter
            q_filt = self.filters_q[h](q_h, edge_index)
            k_filt = self.filters_k[h](k_h, edge_index)
            
            q_spec_list.append(q_filt)
            k_spec_list.append(k_filt)
        
        # Re-stack: [N, H, D]
        q_spec = torch.stack(q_spec_list, dim=1)
        k_spec = torch.stack(k_spec_list, dim=1)
        
        # 3. Convert to dense batch for attention computation
        # This handles variable-sized graphs in a batch
        q_dense, mask = to_dense_batch(q_spec.reshape(x.size(0), -1), batch)
        k_dense, _ = to_dense_batch(k_spec.reshape(x.size(0), -1), batch)
        v_dense, _ = to_dense_batch(v.reshape(x.size(0), -1), batch)
        
        # Reshape for multi-head attention: [B, N, H, D_head] -> [B, H, N, D_head]
        B, max_N, _ = q_dense.size()
        q_dense = q_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        k_dense = k_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        v_dense = v_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_dense, k_dense.transpose(-2, -1)) * scale
        
        # Apply padding mask
        mask_broadcast = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~mask_broadcast, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out_dense = torch.matmul(attn, v_dense)  # [B, H, N, D]
        
        # Reshape back to [B, N, H*D]
        out_dense = out_dense.transpose(1, 2).reshape(B, max_N, self.embed_dim)
        
        # Extract valid nodes (remove padding)
        out = out_dense[mask]  # [Total_N, embed_dim]
        
        # Output projection
        out = self.W_o(out)
        
        # Residual connection and normalization
        x = residual + self.dropout(out)
        
        # Feed-forward network with residual
        x = x + self.ffn(self.norm2(x))
        
        return x
