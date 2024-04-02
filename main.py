import argparse
from pathlib import Path

from src.dataset.retrieve_dataset import get_dataset
from src.train_model import train
from src.translate import translate


def parse_args():
    parser = argparse.ArgumentParser(description="Train or translate with a transformer model")

    # Group for common arguments
    common_group = parser.add_argument_group('Common Arguments')
    common_group.add_argument('-s', '--src_lang', default='en',
                              help="Source language (default: 'en')")
    common_group.add_argument('-t', '--tgt_lang', default='fr',
                              help="Target language (default: 'fr')")
    common_group.add_argument('-mp', '--model_path', default=None,
                              help="Path to the preloaded model weights")

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train mode')
    train_parser.add_argument('-p', '--preload', action='store_true',
                              help="Whether to preload a model for training")

    # Translate mode
    translate_parser = subparsers.add_parser('translate', help='Translate mode')
    translate_parser.add_argument('text', help="Text to translate")

    return parser.parse_args()


def main():
    args = parse_args()

    model_folder = f"models/model_{args.src_lang}_{args.tgt_lang}"
    model_folder_path = Path(model_folder)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    if args.mode == 'translate':
        try:
            weights_path = args.model_path if args.model_path else str(max(model_folder_path.glob("weights/weights_*.pt")))
            translate(args.text, model_folder, weights_path)
        except ValueError:
            print("Error : No weights files found. Please add one or train the model before")
    else:
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model_folder = get_dataset(model_folder)
        if args.preload:
            weights_path = args.model_path if args.model_path else str(max(model_folder_path.glob("weights/weights_*.pt")))
            preload_path = weights_path
        else:
            preload_path = None
        train(model_folder, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, preload_path)


if __name__ == "__main__":
    main()
