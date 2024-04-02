import argparse
from pathlib import Path
from tokenizers import Tokenizer

from src.dataset.retrieve_dataset import get_dataset
from src.model.transformer import Transformer
from src.train_model import train
from src.translate import translate
from config import transformer_parameters, training_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train or translate with a transformer model")

    # Group for common arguments
    common_group = parser.add_argument_group('Common Arguments')
    common_group.add_argument('-s', '--src_lang', default='en',
                              help="Source language (default: 'en')")
    common_group.add_argument('-t', '--tgt_lang', default='fr',
                              help="Target language (default: 'fr')")

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train mode')
    train_parser.add_argument('-p', '--preload', nargs='?', const="latest",
                              help="Whether to preload a model for training and path to the preloaded model weights")

    # Translate mode
    translate_parser = subparsers.add_parser('translate', help='Translate mode')
    translate_parser.add_argument('text', help="Text to translate")
    translate_parser.add_argument('-m', '--model_path', default=None,
                                  help="Path to the preloaded model weights")

    return parser.parse_args()


def main():
    args = parse_args()

    model_folder = f"models/model_{args.src_lang}_{args.tgt_lang}"
    model_folder_path = Path(model_folder)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    device = training_parameters['device']

    if args.mode == 'translate':
        # Get languages and tokenizers
        lang_src, lang_tgt = model_folder.split("_")[1:]

        tokenizer_src_path = Path(f"{model_folder}/tokenizer_{lang_src}")
        tokenizer_tgt_path = Path(f"{model_folder}/tokenizer_{lang_tgt}")

        if not Path.exists(tokenizer_src_path) or not Path.exists(tokenizer_tgt_path):
            raise Exception("Model folder is not found.")
        else:
            tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
            tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

            model = Transformer(
                tokenizer_src.get_vocab_size(),
                tokenizer_tgt.get_vocab_size(),
                transformer_parameters['n_layers'],
                transformer_parameters['d_model'],
                training_parameters['dropout_rate'],
                transformer_parameters['max_seq_length'],
                transformer_parameters['n_head'],
                device
            ).to(device)

            try:
                weights_path = args.model_path if args.model_path \
                    else str(max(model_folder_path.glob("weights/weights_*.pt")))
                translate(args.text, model, tokenizer_src, tokenizer_tgt, weights_path,
                          transformer_parameters['max_seq_length'], device)
            except ValueError:
                print("Error : No weights files found. Please add one or train the model before")
    else:
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model_folder = \
            get_dataset(model_folder, transformer_parameters['max_seq_length'], training_parameters['batch_size'])

        max_seq_length = max(tokenizer_src.get_vocab_size(), tokenizer_src.get_vocab_size()) + 3

        model = Transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            transformer_parameters['n_layers'],
            transformer_parameters['d_model'],
            training_parameters['dropout_rate'],
            max_seq_length,
            transformer_parameters['n_head'],
            device
        ).to(device)

        # if args.preload is not None we load a pre-trained model
        if args.preload is not None:
            try:
                if args.preload == "latest":  # if the model path is not specified, we take the latest
                    preload_path = str(max(model_folder_path.glob("weights/weights_*.pt")))
                else:
                    preload_path = args.preload
            except ValueError:
                print("Error : No weights files found for preloading the model.")
                return
        else:
            preload_path = None
        train(model, model_folder, train_dataloader, val_dataloader,
              tokenizer_src, tokenizer_tgt, training_parameters, preload_path)


if __name__ == "__main__":
    main()
