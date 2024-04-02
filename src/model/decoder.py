import torch
from torch import nn
from .multi_head_attention import MultiHeadAttention
from .encoder import Encoder


# B = Batch_size
# S = Sequence length
# E = Embedding dimension
class Decoder(nn.Module):
    """
    Decoder module for Transformer model

    Args:
        d_model: Dimension of the embedding of the model

    Attributes:
        multi_head_attention_layer (MultiHeadAttention): Multi-head attention layer
        ln1 (torch.nn.LayerNorm): Layer normalization layer
        encoder (Encoder): Encoder module

    Methods:
        forward(q, k, v, encoder_mask, decoder_mask): Forward pass of the Decoder module
    """
    def __init__(self, d_model: int, dropout_rate: float, n_head: int):
        super().__init__()
        self.multi_head_attention_layer = MultiHeadAttention(d_model, dropout_rate, n_head)
        self.ln1 = nn.LayerNorm(d_model)

        self.encoder = Encoder(d_model, dropout_rate, n_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        """
        Forward pass of the Decoder module.

        Args:
            q (torch.Tensor): Query tensor of shape (B, S, E) (from output embedding)
            k (torch.Tensor): Key tensor of shape (B, S, E) (from encoder)
            v (torch.Tensor): Value tensor of shape (B, S, E) (from encoder)
            encoder_mask (torch.Tensor): Encoder mask tensor of shape (B, 1, 1, S)
            decoder_mask (torch.Tensor): Decoder mask tensor of shape (B, 1, S, S)

        Returns:
            torch.Tensor: Output tensor of shape (B, S, E).
        """
        # (B, S, E) -> (B, S, E)
        x = self.multi_head_attention_layer(q, q, q, decoder_mask)
        x = q + self.ln1(x)

        # (B, S, E) -> (B, S, E)
        out = self.encoder(x, k, v, encoder_mask)

        return out  # (B, S, E)
