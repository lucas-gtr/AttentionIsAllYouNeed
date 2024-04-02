import torch
from torch import nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


# B = Batch_size
# S = Sequence length
# E = Embedding dimension
class Encoder(nn.Module):
    """
    Encoder module for Transformer model

    Args:
        d_model: Dimension of the embedding of the model
        dropout_rate: Probability for the dropout layers
        n_head: Number of head

    Attributes:
        multi_head_attention_layer (MultiHeadAttention): Multi-head attention layer
        ln1 (torch.nn.LayerNorm): Layer normalization layer
        ffwd (FeedForward): Feed Forward neural network
        ln2 (torch.nn.LayerNorm): Layer normalization layer

    Methods:
        forward(q, k, v, mask): Forward pass of the Encoder module
    """
    def __init__(self, d_model: int, dropout_rate: float, n_head: int):
        super().__init__()
        self.multi_head_attention_layer = MultiHeadAttention(d_model, dropout_rate, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffwd = FeedForward(d_model, dropout_rate)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass of the Encoder module

        Args:
            q (torch.Tensor): Query tensor of shape (B, S, E)
            k (torch.Tensor): Key tensor of shape (B, S, E)
            v (torch.Tensor): Value tensor of shape (B, S, E)
            mask (torch.Tensor): Mask tensor of shape (B, 1, 1, S)

        Returns:
            torch.Tensor: Output tensor of shape (B, S, E)
        """
        # (B, S, E) -> (B, S, E)
        x1 = self.multi_head_attention_layer(q, k, v, mask)
        x1 = self.dropout1(self.ln1(x1))
        x1 = q + x1

        # (B, S, E) -> (B, S, E)
        x2 = self.ffwd(x1)
        x2 = self.dropout2(self.ln2(x2))
        out = x1 + x2

        return out  # (B, S, E)
