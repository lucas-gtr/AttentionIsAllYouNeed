from pathlib import Path

import torch
from tokenizers import Tokenizer
from .config import max_seq_length, model_device
from .model.transformer import Transformer


def translate(input_sequence, model_folder, model_file):
    # Get languages and tokenizers
    lang_src, lang_tgt = model_folder.split("_")[1:]

    tokenizer_src_path = Path(f"{model_folder}/tokenizer_{lang_src}")
    tokenizer_tgt_path = Path(f"{model_folder}/tokenizer_{lang_tgt}")

    if not Path.exists(tokenizer_src_path) or not Path.exists(tokenizer_tgt_path):
        raise Exception("Model folder is not found.")
    else:
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Prepare sequence and its mask for the model
    sequence_input_tokens = tokenizer_src.encode(input_sequence).ids
    sequence_n_padding = max_seq_length - len(sequence_input_tokens) - 2
    sequence_input = torch.cat(
        [
            sos_token,
            torch.tensor(sequence_input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * sequence_n_padding, dtype=torch.int64)
        ]
    )
    sequence_mask = (sequence_input != pad_token).unsqueeze(0).unsqueeze(0).int()

    # Prepare the model
    model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(model_device)
    model_file_path = Path(model_file)
    state = torch.load(model_file_path, map_location=torch.device(model_device))
    model.load_state_dict(state["model_state_dict"])

    # Calculate the output of the sequence with the model
    sequence_output_tokens = model.generate(sequence_input.to(model_device), sequence_mask.to(model_device),
                                            sos_idx, eos_idx)
    # Convert the output into string
    sequence_translated = tokenizer_tgt.decode(sequence_output_tokens.detach().cpu().numpy())

    print(f'SOURCE: {input_sequence}')
    print(f'TRANSLATION: {sequence_translated}')
