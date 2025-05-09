from torch.utils.data import DataLoader
from .dataset import MapsDataset
from ..config import default_config

def get_dataloaders(
    root_dir: str = None, 
    batch_size: int = None, 
    target_shape: tuple = None, 
    num_workers: int = None, 
    test_size: float = None, 
    seed: int = None
):
    # Use config values as defaults
    config_dataset = default_config['dataset']
    config_loader = default_config['dataloader']

    root_dir = root_dir or config_dataset['root_dir']
    target_shape = target_shape or config_dataset['target_shape']
    test_size = test_size or config_dataset['test_size']
    seed = seed or config_dataset['seed']

    batch_size = batch_size or config_loader['batch_size']
    num_workers = num_workers or config_loader['num_workers']

    # Create datasets
    train_dataset = MapsDataset(
        root_dir=root_dir,
        target_shape=target_shape,
        is_train=True,
        test_size=test_size
    )

    val_dataset = MapsDataset(
        root_dir=root_dir,
        target_shape=target_shape,
        is_train=False,
        test_size=test_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader