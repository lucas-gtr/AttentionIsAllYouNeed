import torch
import math

from src.config import max_seq_length, d_model, model_device


# S = Sequence_length
# E = Embedding dimension
def get_positional_encoding():
    """
    Generate positional encodings.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (1, S, E).
    """
    # (S, E)
    pe = torch.zeros(max_seq_length, d_model)

    # (S, 1)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    # (E // 2)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    # (S, E)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # (S, E) -> (1, S, E)
    pe = pe.unsqueeze(0)

    return pe.to(model_device)  # (1, S, E)
