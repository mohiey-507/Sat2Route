import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from sat2route.config import default_config
from sat2route.models import Generator, Discriminator
from sat2route.losses.loss import Loss
from sat2route.data.data_loader import get_dataloaders

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    return logger

class Trainer:
    def __init__(
        self,
        generator: nn.Module = None,
        discriminator: nn.Module = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        gen_optimizer: optim.Optimizer = None,
        disc_optimizer: optim.Optimizer = None,
        loss_fn: nn.Module = None,
        device: torch.device = None,
        log_name: str = 'trainer',
        checkpoint_dir: str = 'checkpoints',
        display_step: int = 200,
    ):
        # Initialize logger
        self.logger = get_logger(log_name)
        self.logger.info(f"Initializing trainer")

        # Get training config
        self.training_config = default_config['training']
        self.lambda_recon = self.training_config['lambda_recon']
        self.epochs = self.training_config['epochs']
        self.lr = self.training_config['lr']
        self.beta1 = self.training_config['beta1']
        self.beta2 = self.training_config['beta2']
        
        self.logger.info(f"Training config: {self.training_config}")

        # Set device
        self.device = device if device is not None else torch.device(
            default_config['training']['device']
        )

        # Get model configs
        self.generator_config = default_config['model']['generator']
        self.discriminator_config = default_config['model']['discriminator']
        
        in_channels = default_config['model']['generator']['in_channels']
        out_channels = default_config['model']['generator']['out_channels']
        g_hidden_channels = default_config['model']['generator']['hidden_channels']
        g_depth = default_config['model']['generator']['depth']
        
        d_in_channels = default_config['model']['discriminator']['in_channels']
        d_hidden_channels = default_config['model']['discriminator']['hidden_channels']
        d_depth = default_config['model']['discriminator']['depth']
        
        self.generator = generator if generator is not None else Generator(
            in_ch=in_channels, 
            out_ch=out_channels, 
            hidden_ch=g_hidden_channels, 
            depth=g_depth
        )
        
        self.discriminator = discriminator if discriminator is not None else Discriminator(
            in_ch=d_in_channels, 
            hidden_ch=d_hidden_channels, 
            depth=d_depth
        )

        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize optimizers using config parameters
        self.gen_optimizer = gen_optimizer if gen_optimizer is not None else \
            optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.disc_optimizer = disc_optimizer if disc_optimizer is not None else \
            optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # Initialize loss function
        self.loss_fn = loss_fn if loss_fn is not None else Loss(lambda_recon=self.lambda_recon).to(self.device)
        
        # Initialize data loaders if not provided
        if train_loader is None or val_loader is None:
            self.train_loader, self.val_loader = get_dataloaders()
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader
        
        # Setup checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"Checkpoint directory: {os.path.abspath(checkpoint_dir)}")

        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.cur_step = 0
        self.display_step = display_step

    def train_step(self, batch) -> dict:
        """Execute a single training step (generator + discriminator) on a batch.
        
        Args:
            batch: Tuple of (condition, target) from dataloader
                condition: Satellite image [B, 3, H, W]
                target: Map image [B, 3, H, W]
                
    Returns:
            dict: Dictionary with all loss components
        """
        condition, target = batch
        condition = condition.to(self.device)
        target = target.to(self.device)
        
        # Train discriminator
        self.disc_optimizer.zero_grad()
        disc_losses = self.loss_fn(self.discriminator, target, condition, mode='discriminator', gen=self.generator)
        disc_losses['total'].backward()
        self.disc_optimizer.step()
        
        # Train generator
        self.gen_optimizer.zero_grad()
        gen_losses = self.loss_fn(self.generator, target, condition, mode='generator', disc=self.discriminator)
        gen_losses['total'].backward()
        self.gen_optimizer.step()
        
        # Generate fake image for visualization
        with torch.no_grad():
            fake = self.generator(condition)
        
        # Visualization
        if self.display_step > 0 and self.cur_step % self.display_step == 0:
            input_dim = condition.shape[1]
            real_dim = target.shape[1]
            target_shape = condition.shape[2]
            print(f"\nStep {self.cur_step}: Visualization of condition (input), target (real), and generated (fake) images")
            
            self.show_tensor_images(condition, num_images=4, size=(input_dim, target_shape, target_shape))
            
            self.show_tensor_images(target, num_images=4, size=(real_dim, target_shape, target_shape))
            
            self.show_tensor_images(fake, num_images=4, size=(real_dim, target_shape, target_shape))
        
        # Increment step counter
        self.cur_step += 1
        
        return {
            'disc': disc_losses,
            'gen': gen_losses
        }