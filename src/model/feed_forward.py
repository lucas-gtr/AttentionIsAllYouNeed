import torch
from torch import nn
from src.config import d_model, dropout_rate


# B = Batch_size
# S = Sequence length
# E = Embedding dimension
class FeedForward(nn.Module):
    """
    Feed Forward Neural Network module for Transformer model

    Attributes:
        ffn (torch.nn.Sequential): Sequential neural network layers

    Methods:
        forward(x): Forward pass of the Feed Forward module
    """
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            # (B, S, E) -> -(B, S, 4*E)
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            # (B, S, 4*E) -> -(B, S, E)
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Feed Forward module

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, E)

        Returns:
            torch.Tensor: Output tensor of shape (B, S, E)
        """
        out = self.ffn(x)

        return out
