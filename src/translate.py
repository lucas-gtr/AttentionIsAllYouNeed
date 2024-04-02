from pathlib import Path

import torch


def translate(input_sequence, model, tokenizer_src, tokenizer_tgt, model_file, max_seq_length, device):
    """
    Translate the input sequence using the provided model and tokenizers

    Args:
        input_sequence (str): The input sequence to be translated
        model (Transformer): The trained model used for translation
        tokenizer_src: Tokenizer for the source language
        tokenizer_tgt: Tokenizer for the target language
        model_file (str): Path to the file containing the trained model weights
        max_seq_length (int): Maximum length of the input sequence
        device (str): Device to run the translation on
    """

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
    model_file_path = Path(model_file)
    state = torch.load(model_file_path, map_location=torch.device(device))
    model.load_state_dict(state["model_state_dict"])

    # Calculate the output of the sequence with the model
    sequence_output_tokens = model.generate(sequence_input.to(device), sequence_mask.to(device),
                                            sos_idx, eos_idx)
    # Convert the output into string
    sequence_translated = tokenizer_tgt.decode(sequence_output_tokens.detach().cpu().numpy())

    print(f'SOURCE: {input_sequence}')
    print(f'TRANSLATION: {sequence_translated}')
