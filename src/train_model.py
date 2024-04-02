import torch
from torch import nn
from pathlib import Path

from .model.transformer import Transformer

from tqdm import tqdm


def train(model, model_folder, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt,
          training_parameters, preload=None):
    """
    Train the provided model using the given data loaders and parameters

    Args:
        model (Transformer): The model to be trained
        model_folder (str): Path to the folder where model weights will be saved
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data
        tokenizer_src: Tokenizer for the source language
        tokenizer_tgt: Tokenizer for the target language
        training_parameters (dict): Dictionary containing training parameters
            'device' (str): Device (CPU or GPU) to run the training on
            'lr' (float): Learning rate for the optimizer
            'beta_1' (float): Beta1 parameter for Adam optimizer
            'beta_2' (float): Beta2 parameter for Adam optimizer
            'epsilon' (float): Epsilon parameter for Adam optimizer
            'epochs' (int): Number of epochs to train for
        preload (str, optional): Path to a pretrained model to resume training from
    """
    device = training_parameters['device']
    print(f"Using device {device}")
    device = torch.device(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=training_parameters['lr'],
                                 betas=(training_parameters['beta_1'], training_parameters['beta_2']),
                                 eps=training_parameters['epsilon'])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    initial_epoch = 0
    if preload:
        print(f"Preloading model {preload}")
        state = torch.load(Path(preload))
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        model.load_state_dict(state["model_state_dict"])

    for epoch in range(initial_epoch, training_parameters['epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        model.train()
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)  # (B, S)
            decoder_input = batch['decoder_input'].to(device)  # (B, S)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, S)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, S, S)

            logits = model(encoder_input, decoder_input, encoder_mask, decoder_mask)  # (B, S, V)

            label = batch['label'].to(device)

            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        weight_folder = Path(f"{model_folder}/weights")
        Path(weight_folder).mkdir(parents=True, exist_ok=True)
        save_model(epoch, model, optimizer, weight_folder)

        run_validation(model, val_dataloader, tokenizer_tgt, lambda msg: batch_iterator.write(msg),
                       device, num_examples=5)


def save_model(epoch, model, optimizer, weight_folder):
    model_file_name = Path(f"{weight_folder}/weights_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_file_name)


def run_validation(model: Transformer, validation_dataset, tokenizer_tgt, print_msg, device, num_examples=2):
    """
    Run validation on the provided dataset using the given model

    Args:
        model (Transformer): Transformer model
        validation_dataset: Validation dataset
        tokenizer_tgt: Tokenizer for target language
        print_msg (callable): Function for printing messages
        device: Device (CPU or GPU) to run the training on
        num_examples (int): Number of examples to print
    """
    model.eval()

    count = 0

    console_width = 80

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            predicted_tokens = model.generate(encoder_input, encoder_mask, sos_idx, eos_idx)

            source_text = batch["src_text"][0]
            expected_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(predicted_tokens.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {expected_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
