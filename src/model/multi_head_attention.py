import math

import torch
from torch import nn
from src.config import d_model, n_head, dropout_rate


# B = Batch_size
# S = Sequence length
# E = Embedding dimension
# N = Number of head
# H = Head dimension
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for Transformer model

    Attributes:
        d_k (int): Dimension of each head.
        w_q (torch.nn.Linear): Linear layer for query
        w_k (torch.nn.Linear): Linear layer for key
        w_v (torch.nn.Linear): Linear layer for value
        w_o (torch.nn.Linear): Linear layer for output
        dropout (torch.nn.Dropout): Dropout layer for regularization

    Methods:
        attention(q, k, v, mask): Perform scaled dot-product attention
        forward(q, k, v, mask): Forward pass of the Multi-Head Attention module
    """
    def __init__(self):
        super().__init__()
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        """
        Perform scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (B, N, S, H)
            k (torch.Tensor): Key tensor of shape (B, N, S, H)
            v (torch.Tensor): Value tensor of shape (B, N, S, H)
            mask (torch.Tensor): Mask tensor of shape (B, 1, 1, S) (encoder) or (B, 1, S, S) (decoder)

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H)
        """
        # (B, N, S, H) -> (B, N, H, S)
        kT = k.transpose(-2, -1)
        # (B, N, S, H) @ (B, N, H, S) -> (B, N, S, S)
        attention_weights = (q @ kT) / math.sqrt(self.d_k)
        if mask is not None:
            attention_weights.masked_fill_(mask == 0, float('-inf'))
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        # (B, N, S, S) @ (B, N, S, H) -> (B, N, S, H)
        out = attention_weights @ v

        return out  # (B, N, S, H)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass of the Multi-Head Attention module

        Args:
            q (torch.Tensor): Query tensor of shape (B, S, E)
            k (torch.Tensor): Key tensor of shape (B, S, E)
            v (torch.Tensor): Value tensor of shape (B, S, E)
            mask (torch.Tensor): Mask tensor of shape (B, S, S)

        Returns:
            torch.Tensor: Output tensor of shape (B, S, E)
        """

        # (B, S, E) -> (B, S, E)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (B, S, E) -> (B, S, N, H) -> (B, N, S, H)
        query = query.view(query.shape[0], query.shape[1], n_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], n_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], n_head, self.d_k).transpose(1, 2)

        multi_head_attention = self.attention(query, key, value, mask)  # (B, N, S, H)

        # (B, N, S, H) -> (B, S, N, H) -> (B, S, E)
        multi_head_attention = multi_head_attention.transpose(1, 2).contiguous().view(multi_head_attention.shape[0], -1,
                                                                                      n_head * self.d_k)

        # (B, S, E) -> (B, S, E)
        out = self.w_o(multi_head_attention)

        return out  # (B, S, E)
