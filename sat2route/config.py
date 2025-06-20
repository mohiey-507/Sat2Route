import os
import torch

class Config:
    """
    Manages all configuration parameters for datasets, models, and training.
    """
    def __init__(self):
        # Default configuration
        self.config = {
            # Dataset parameters
            'dataset': {
                'root_dir': 'sat2route/datasets/maps',
                'target_shape': (256, 256),
                'test_size': 0.2,
                'seed': 7
            },
            # Dataloader parameters
            'dataloader': {
                'batch_size': 16,
                'num_workers': min(4, os.cpu_count() or 1),
            },
            # Model parameters
            'model': {
                'generator': {
                    'in_channels': 3,
                    'out_channels': 3,
                    'hidden_channels': 32,
                    'depth': 6,
                    'use_dropout': True
                },
                'discriminator': {
                    'in_channels': 6,
                    'hidden_channels': 8,
                    'depth': 4
                }
            },
            # Training parameters
            'training': {
                'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                'lambda_recon': 200.0,
                'epochs': 2,
                'lr': 0.0002,
                'beta1': 0.5,
                'beta2': 0.999,
            }
        }
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def get(self, key, default=None):
        return self.config.get(key, default)

default_config = Config()