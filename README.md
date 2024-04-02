# Attention is All You Need / PyTorch

## Description
This repository presents an implementation of the [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) paper using Python and PyTorch.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Training](#training)
- [Translation](#translation)

## Overview
This project implements the Transformer model architecture proposed in the 'Attention is all you need' paper. Its primary functionalities include:
* Training a model on the 'Opus books' dataset in a user-specified language.
* Utilizing a trained model for translating sentences into another language.

## Installation
To use this project, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage
The project offers functionalities for both training Transformer models on bilingual datasets and performing translations.

### To train a model:
To train a model, use the following command:
```bash
python main.py train [--src_lang SRC_LANG] [--tgt_lang TGT_LANG] [--preload MODEL_PATH]
```

* Use `-s` or `--src_lang`  to specify the source language ('en' by default).
* Use `-t` or `--tgt_lang`  to specify the target language ('fr' by default).
* `-p` or `--preload` to preload a model for training. If **MODEL_PATH** is not specified, the latest model is used.

### To translate text:
To translate text, execute the following command:
```bash
python main.py translate TEXT [--src_lang SRC_LANG] [--tgt_lang TGT_LANG] [--model_path MODEL_PATH]
```

* `TEXT` refers to the text to be translated.
* Use `-s` or `--src_lang`  to specify the source language ('en' by default).
* Use `-t` or `--tgt_lang`  to specify the target language ('fr' by default).
* `-mp` or `--model_path` specifies the path to the preloaded model weights. If **MODEL_PATH** is not specified, the latest model is used.

## Project organization
The [config file](config.py) contains the model and training parameters. 
Model weights and tokenizers are saved in the [models folder](models/) according to the language pair (e.g., model_en_fr for English to French translation).

## Model

The Transformer model architecture, as described in the 'Attention is all you need' paper, comprises encoder and decoder layers, integrating multi-head attention mechanisms with feed-forward neural networks. Model parameters include: 
* `n_layers`: Number of layers for both encoder and decoder.
* `max_seq_length`: Maximum token length for inputs, automatically adjusted during training based on the dataset's longest sentence.
* `d_model`: Dimension of word embeddings.
* `n_head`: Number of head for multi-head attention.

These parameters can be adjusted in the [configuration file](config.py)

## Training

The training process employs the Adam Optimizer to optimize Transformer model parameters on bilingual datasets. Training parameters include: 
* `device`:  Device for model execution (GPU if available, otherwise CPU).
* `batch_size`: Batch size for training.
* `epochs`: Number of training epochs.
* `beta_1`, `beta_2`, `epsilon`: Parameters for the Adam Optimizer.
* `lr`: Learning rate.
* `dropout_rate`: Dropout rate for model dropout layers.

These parameters can be adjusted in the [configuration file](config.py)

## Translation

For translation tasks, the trained Transformer model is utilized to translate text from the source language to the target language. Pre-trained model weights must be loaded for translation tasks.
