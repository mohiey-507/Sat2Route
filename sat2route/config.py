import torch

DEFAULT_CONFIG = {
    # Dataset parameters
    'dataset': {
        'root_dir': 'sat2route/datasets/maps',
        'target_shape': (256, 256),
        'test_size': 0.2,
        'seed': 7
    },
    # Dataloader parameters
    'dataloader': {
        'batch_size': 4,
        'num_workers': 4,
    },
    # Model parameters
    'model': {
        'generator': {
            'in_channels': 3,
            'out_channels': 3,
            'hidden_channels': 64,
            'depth': 7,
            'use_dropout': True
        },
        'discriminator': {
            'in_channels': 6,
            'hidden_channels': 64,
            'depth': 5
        }
    },
    # Training parameters
    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lambda_recon': 200.0,
        'epochs': 10,
        'lr': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
    }
}

def get_config(override_dict: dict = None) -> dict:
    config = DEFAULT_CONFIG.copy()

    if override_dict:
        config.update({k: override_dict[k] for k in override_dict if override_dict[k] is not None})

    return config