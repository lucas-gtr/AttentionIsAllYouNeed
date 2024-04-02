from torch.utils.data import DataLoader, random_split
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from src.config import batch_size, max_seq_length
from .bilingual_dataset import BilingualDataset


def get_all_sentences(dataset, lang):
    """
    Generator function to yield sentences from the dataset

    Args:
        dataset: Dataset object
        lang (str): Language of the sequence

    Yields:
        str: Sentence from the dataset in the correct language
    """
    for item in dataset:
        yield item['translation'][lang]


def get_tokenizer(dataset, model_folder, lang):
    """
    Get or create tokenizer for a specific language

    Args:
        dataset: Dataset object
        model_folder: Path to the model folder
        lang (str): Language of the tokenizer

    Returns:
        Tokenizer: Tokenizer object for the language
    """
    tokenizer_path = Path(f"{model_folder}/tokenizer_{lang}")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(model_folder):
    """
    Get training and validation datasets

    Returns:
        DataLoader: DataLoader object for training dataset
        DataLoader: DataLoader object for validation dataset
        Tokenizer: Tokenizer object for source language
        Tokenizer: Tokenizer object for target language
    """
    lang_src, lang_tgt = model_folder.split("_")[1:]

    dataset_raw = load_dataset('opus_books', f'{lang_src}-{lang_tgt}', split='train')

    tokenizer_src = get_tokenizer(dataset_raw, model_folder, lang_src)
    tokenizer_tgt = get_tokenizer(dataset_raw, model_folder, lang_tgt)

    train_size = int(0.9 * len(dataset_raw))
    val_size = len(dataset_raw) - train_size

    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_size, val_size])

    train_dataset = BilingualDataset(train_dataset_raw, lang_src, tokenizer_src, lang_tgt, tokenizer_tgt)
    val_dataset = BilingualDataset(val_dataset_raw, lang_src, tokenizer_src, lang_tgt, tokenizer_tgt)

    max_len_src = max(len(tokenizer_src.encode(item['translation'][lang_src]).ids) for item in dataset_raw)
    max_len_tgt = max(len(tokenizer_tgt.encode(item['translation'][lang_tgt]).ids) for item in dataset_raw)
    assert max_len_tgt + 2 < max_seq_length or max_len_src + 2 < max_seq_length, \
        "There are sentences longer than the maximum defined in config, increase the value in config"

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model_folder
