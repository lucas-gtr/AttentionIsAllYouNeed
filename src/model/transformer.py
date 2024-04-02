import math

import torch
from torch import nn
from .positional_encoding import get_positional_encoding
from .encoder import Encoder
from .decoder import Decoder


# B = Batch_size
# S = Sequence length
# E = Embedding dimension
# V = Vocabulary size
class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks

    Args:
        vocab_src_size (int): Size of the source vocabulary
        vocab_tgt_size (int): Size of the target vocabulary
        n_layers (int) : Number of layers for encoders and decoders
        d_model (int) : Dimension of the embedding of the model
        dropout_rate (float) : Probability for the dropout layers
        max_seq_length (int) : Maximum token length for a sequence
        n_head (int) : Number of heads
        device (str) : Device to run the model on

    Attributes:
        pe (torch.Tensor): Positional encoding tensor
        encoder_embedding (torch.nn.Embedding): Embedding layer for the source vocabulary
        dropout_encoder (torch.nn.Dropout): Dropout layer for encoder inputs
        encoders (torch.nn.ModuleList): List of encoder layers
        encoder_norm (torch.nn.LayerNorm): Layer normalization layer for encoder output
        decoder_embedding (torch.nn.Embedding): Embedding layer for the target vocabulary
        dropout_decoder (torch.nn.Dropout): Dropout layer for decoder inputs
        decoders (torch.nn.ModuleList): List of decoder layers
        decoder_norm (torch.nn.LayerNorm): Layer normalization layer for decoder output
        fn (torch.nn.Linear): Linear layer for final output

    Methods:
        encode(inputs, encoder_mask): Encode input sequence
        decode(decoder_input, encoder_output, encoder_mask, decoder_mask): Decode output sequence
        forward(inputs, targets, encoder_mask=None, decoder_mask=None): Forward pass of the Transformer model
        generate(inputs, encoder_mask, sos_idx, eos_idx): Generate output sequence given input sequence
    """

    def __init__(self, vocab_src_size: int, vocab_tgt_size: int, n_layers: int, d_model: int,
                 dropout_rate: float, max_seq_length: int, n_head: int, device: str):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.device = device

        self.pe = get_positional_encoding(d_model, max_seq_length, device)  # (1, S, D)

        self.encoder_embedding = nn.Embedding(vocab_src_size, d_model)
        self.dropout_encoder = nn.Dropout(dropout_rate)
        self.encoders = nn.ModuleList([Encoder(d_model, dropout_rate, n_head) for _ in range(n_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_embedding = nn.Embedding(vocab_tgt_size, d_model)
        self.dropout_decoder = nn.Dropout(dropout_rate)
        self.decoders = nn.ModuleList([Decoder(d_model, dropout_rate, n_head) for _ in range(n_layers)])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.fn = nn.Linear(d_model, vocab_tgt_size)

    def encode(self, inputs, encoder_mask):
        """
        Encode input sequence

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, S)
            encoder_mask (torch.Tensor): Encoder mask tensor of shape (B, 1, 1, S)

        Returns:
            torch.Tensor: Encoder output tensor of shape (B, S, E)
        """
        # (B, S) -> (B, S, E)
        x = self.encoder_embedding(inputs) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        x = self.dropout_encoder(x)

        # (B, S, E) -> (B, S, E)
        for encoder in self.encoders:
            x = encoder(x, x, x, encoder_mask)
        x = self.encoder_norm(x)

        return x  # (B, S, E)

    def decode(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        """
        Decode output sequence

        Args:
            decoder_input (torch.Tensor): Decoder input tensor of shape (B, S)
            encoder_output (torch.Tensor): Encoder output tensor of shape (B, S, E)
            encoder_mask (torch.Tensor): Encoder mask tensor of shape (B, 1, 1, S)
            decoder_mask (torch.Tensor): Decoder mask tensor of shape (B, 1, S, S)

        Returns:
            torch.Tensor: Decoder output tensor of shape (B, S, E)
        """
        # (B, S) -> (B, S, E)
        x = self.decoder_embedding(decoder_input) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        x = self.dropout_decoder(x)

        # (B, S, E) -> (B, S, E)
        for decoder in self.decoders:
            x = decoder(x, encoder_output, encoder_output, encoder_mask, decoder_mask)
        out = self.decoder_norm(x)

        return out  # (B, S, E)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                encoder_mask: torch.Tensor, decoder_mask: torch.Tensor):
        """
        Forward pass of the Transformer model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, S)
            targets (torch.Tensor): Target tensor of shape (B, S)
            encoder_mask (torch.Tensor): Encoder mask tensor of shape (B, 1, 1, S)
            decoder_mask (torch.Tensor): Decoder mask tensor of shape (B, 1, S, S)

        Returns:
            torch.Tensor: Output tensor of shape (B, S, V).
        """
        encoder_output = self.encode(inputs, encoder_mask)  # (B, S, E)
        decoder_output = self.decode(targets, encoder_output, encoder_mask, decoder_mask)  # (B, S, E)
        # (B, S, E) -> (B, S, V)
        logits = self.fn(decoder_output)
        return logits  # (B, S, V)

    # current_S = Current sequence length of the generated sequence
    def generate(self, inputs: torch.Tensor, encoder_mask: torch.Tensor, sos_idx, eos_idx):
        """
        Generate output sequence given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor of shape (1, S)
            encoder_mask (torch.Tensor): Encoder mask tensor of shape (1, 1, S)
            sos_idx (int): Index of the start-of-sequence token
            eos_idx (int): Index of the end-of-sequence token

        Returns:
            torch.Tensor: Generated output sequence tensor of shape (S,)
        """
        decoder_input = torch.tensor([[sos_idx]]).type_as(inputs).to(self.device)  # (1, 1)

        encoder_output = self.encode(inputs, encoder_mask)  # (1, S, E)

        while decoder_input.size(1) < self.max_seq_length:
            mask = torch.triu(torch.ones(1, decoder_input.shape[1], decoder_input.shape[1]), diagonal=1).type(torch.int)
            decoder_mask = (mask == 0).type_as(encoder_mask)  # (1, current_S, current_S)

            decoder_output = self.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)  # (1, current_S, E)
            # (1, current_S, E) -> (1, V)
            logits = self.fn(decoder_output[:, -1])

            next_word = torch.argmax(logits, dim=-1)  # (1,)

            # (1, current_S) -> (1, current_S + 1)
            decoder_input = torch.cat((decoder_input, next_word.unsqueeze(0)), dim=1).to(self.device)
            if next_word.item() == eos_idx:
                break

        return decoder_input.squeeze(0)  # (S,)
