import torch
from torch.utils.data import DataLoader
from .dataset import MapsDataset
from ..config import get_config

def get_dataloaders(
    config: dict = None,
    root_dir: str = None, 
    batch_size: int = None, 
    target_shape: tuple = None, 
    num_workers: int = None, 
    test_size: float = None, 
    seed: int = None,
    pin_memory: bool = True
):
    config = config or get_config()
    cfg_ds = config['dataset']
    cfg_dl = config['dataloader']

    root_dir = root_dir or cfg_ds['root_dir']
    target_shape = target_shape or cfg_ds['target_shape']
    test_size = test_size or cfg_ds['test_size']
    seed = seed or cfg_ds['seed']

    batch_size = batch_size or cfg_dl['batch_size']
    num_workers = num_workers or cfg_dl['num_workers']

    g = torch.Generator()
    g.manual_seed(seed)

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
        drop_last=True,
        pin_memory=pin_memory,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(8 * batch_size, 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g
    )

    return train_loader, val_loader