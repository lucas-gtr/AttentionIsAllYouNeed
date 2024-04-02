import torch

# TRANSFORMER PARAMETERS
transformer_parameters = {
    'n_layers': 6,
    'max_seq_length': 485,
    'd_model': 512,
    'n_head': 8
}

assert transformer_parameters['d_model'] % transformer_parameters['n_head'] == 0, \
    "d_model is not divisible by the number of head"

# TRAINING PARAMETERS
training_parameters = {
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
    'batch_size': 8,
    'epochs': 20,
    'beta_1': 0.9,
    'beta_2': 0.98,
    'epsilon': 1e-9,
    'lr': 10**-4,
    'dropout_rate': 0.1
}
