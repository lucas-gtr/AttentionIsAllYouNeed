import torch
import math


# S = Sequence_length
# E = Embedding dimension
def get_positional_encoding(d_model: int, max_seq_length: int, device: str):
    """
    Generate positional encodings.

    Args:
        d_model: Dimension of the embedding of the model
        max_seq_length: Maximum token length for a sequence
        device: Device to run the translation on

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

    return pe.to(device)  # (1, S, E)
