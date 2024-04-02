import torch
from torch.utils.data import Dataset
from src.config import max_seq_length


class BilingualDataset(Dataset):
    """
    Dataset class for handling bilingual data

    Args:
        dataset: Dataset object containing bilingual data
        tokenizer_src: Tokenizer object for source language
        tokenizer_tgt: Tokenizer object for target language

    Attributes:
        dataset: Dataset object containing bilingual data
        tokenizer_src: Tokenizer object for source language
        tokenizer_tgt: Tokenizer object for target language
        sos_token: Start-of-sequence token
        eos_token: End-of-sequence token
        pad_token: Padding token

    Methods:
        __len__(): Returns the number of samples in the dataset
        __getitem__(idx): Retrieves a sample from the dataset
    """

    def __init__(self, dataset, lang_src, tokenizer_src, lang_tgt, tokenizer_tgt):
        self.dataset = dataset

        self.lang_src = lang_src
        self.tokenizer_src = tokenizer_src

        self.lang_tgt = lang_tgt
        self.tokenizer_tgt = tokenizer_tgt

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        """
        Returns the number of samples in the dataset

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset

        Args:
            idx (int): Index of the sample

        Returns:
            dict: A sample from the dataset containing encoder input, decoder input,
                  encoder mask, decoder mask, label, source text, and target text
        """
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_n_padding = max_seq_length - len(enc_input_tokens) - 2
        dec_n_padding = max_seq_length - len(dec_input_tokens) - 1

        if enc_n_padding < 0 or dec_n_padding < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_n_padding, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_n_padding, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_n_padding, dtype=torch.int64)
            ]
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    """
    Generate a causal mask for the decoder

    Args:
        size (int): Size of the mask

    Returns:
        torch.Tensor: Causal mask tensor of shape (1, S, S)
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
