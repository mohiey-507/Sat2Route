import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.amp import GradScaler, autocast

from sat2route.config import default_config
from sat2route.models import Generator, Discriminator
from sat2route.losses.loss import Loss
from sat2route.data.data_loader import get_dataloaders

# Use pathlib for robust path handling
log_dir = Path(__file__).resolve().parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_file = log_dir / f'{name}.log'
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
        
        # Initialize GradScalers for AMP
        self.gen_scaler = GradScaler()
        self.disc_scaler = GradScaler()
        
        # Initialize data loaders if not provided
        if train_loader is None or val_loader is None:
            self.train_loader, self.val_loader = get_dataloaders()
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir.resolve()}")

        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.cur_step = 0
        self.display_step = display_step

    def show_tensor_images(self, image_tensor, num_images=4, size=(3, 256, 256)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_shifted = image_tensor
        image_unflat = image_shifted.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=4)
        plt.figure(figsize=(5, 5))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.axis('off')
        plt.show()

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
        with autocast(device_type=self.device.type):
            disc_losses = self.loss_fn(self.discriminator, target, condition, mode='discriminator', gen=self.generator)
        self.disc_scaler.scale(disc_losses['total']).backward()
        self.disc_scaler.step(self.disc_optimizer)
        self.disc_scaler.update()
        
        # Train generator
        self.gen_optimizer.zero_grad()
        with autocast(device_type=self.device.type):
            gen_losses = self.loss_fn(self.generator, target, condition, mode='generator', disc=self.discriminator)
        self.gen_scaler.scale(gen_losses['total']).backward()
        self.gen_scaler.step(self.gen_optimizer)
        self.gen_scaler.update()
        
        # Generate fake image for visualization
        with torch.inference_mode():
            with autocast(device_type=self.device.type):
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

    def train_epoch(self):
        """Train the model for one epoch.
        
        Returns:
            dict: Dictionary with average losses
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {
            'disc': {'total': 0, 'real': 0, 'fake': 0},
            'gen': {'total': 0, 'adv': 0, 'recon': 0}
        }
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        pbar.set_description(f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        for i, batch in pbar:
            batch_losses = self.train_step(batch)
            
            # Update epoch losses
            for model_type in ['disc', 'gen']:
                for loss_type, loss_value in batch_losses[model_type].items():
                    epoch_losses[model_type][loss_type] += loss_value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{batch_losses['gen']['total'].item():.4f}",
                'D_loss': f"{batch_losses['disc']['total'].item():.4f}"
            })
        
        # Calculate averages
        n_batches = len(self.train_loader)
        for model_type in ['disc', 'gen']:
            for loss_type in epoch_losses[model_type]:
                epoch_losses[model_type][loss_type] /= n_batches
        
        self.logger.info(
            f"Epoch {self.current_epoch+1}/{self.epochs} | "
            f"G_loss: {epoch_losses['gen']['total']:.4f} | "
            f"D_loss: {epoch_losses['disc']['total']:.4f}"
        )
        
        return epoch_losses

    def validate(self):
        """Validate the model on the validation set.
        
        Returns:
            dict: Dictionary with validation losses
            float: Total validation loss
            list: Samples for visualization [condition, target, generated]
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {
            'disc': {'total': 0, 'real': 0, 'fake': 0},
            'gen': {'total': 0, 'adv': 0, 'recon': 0}
        }
        
        samples = []
        
        with torch.inference_mode():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            pbar.set_description("Validation")
            
            for i, batch in pbar:
                condition, target = batch
                condition = condition.to(self.device)
                target = target.to(self.device)
                
                with autocast(device_type=self.device.type):
                    # Generate fake images
                    fake = self.generator(condition)
                    
                    # Calculate discriminator loss
                    disc_losses = self.loss_fn(self.discriminator, target, condition, 
                                        mode='discriminator', gen=self.generator)
                    
                    # Calculate generator loss
                    gen_losses = self.loss_fn(self.generator, target, condition, 
                                        mode='generator', disc=self.discriminator)
                
                # Update validation losses
                for model_type in ['disc', 'gen']:
                    for loss_type, loss_value in (
                        disc_losses.items() if model_type == 'disc' else gen_losses.items()
                    ):
                        if loss_type in val_losses[model_type]:
                            val_losses[model_type][loss_type] += loss_value.item()
                
                # Save first batch for visualization
                if i == 0:
                    samples = [condition[:4], target[:4], fake[:4]]
        
        # Calculate averages
        n_batches = len(self.val_loader)
        for model_type in ['disc', 'gen']:
            for loss_type in val_losses[model_type]:
                val_losses[model_type][loss_type] /= n_batches
        
        # Calculate total loss (generator loss)
        total_val_loss = val_losses['gen']['total']
        
        self.logger.info(
            f"Validation | "
            f"G_loss: {val_losses['gen']['total']:.4f} | "
            f"D_loss: {val_losses['disc']['total']:.4f}"
        )
        
        return val_losses, total_val_loss, samples

    def save_checkpoint(self, epoch, val_loss, is_best=False, is_final=False):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'val_loss': val_loss,
            'config': default_config.config
        }
        
        # Save latest checkpoint
        latest_checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Handle best model
        if is_best:
            best_model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Handle final model
        if is_final:
            final_model_path = self.checkpoint_dir / 'final_model.pth'
            torch.save(checkpoint, final_model_path)
            self.logger.info(f"Final model saved with val_loss: {val_loss:.4f}")
            
            # If the final model is also the best, remove the redundant best_model.pth
            if is_best and 'best_model_path' in locals() and best_model_path.exists():
                best_model_path.unlink()
                self.logger.info("Removed redundant best_model.pth as it is identical to final_model.pth")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        self.best_val_loss = checkpoint['val_loss']
        epoch = checkpoint['epoch']
        
        self.logger.info(f"Loaded checkpoint from epoch {epoch} with val_loss: {self.best_val_loss:.4f}")
        return epoch
    
    def fit(self, epochs=None, resume_from=None):
        """Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train for. If None, use config value.
            resume_from: Checkpoint path to resume from. If None, start from scratch.
            
        Returns:
            dict: Dictionary with training history
        """
        if epochs is not None:
            self.epochs = epochs
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            
        self.current_epoch = start_epoch
        
        history = {
            'train_losses': [],
            'val_losses': [],
        }
        
        self.logger.info(f"Starting training for {self.epochs} epochs")
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            history['train_losses'].append(train_losses)
            
            # Validate
            val_losses, total_val_loss, samples = self.validate()
            history['val_losses'].append(val_losses)
            
            # Determine if this is the best model so far
            is_best = total_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = total_val_loss
            
            # Check if this is the final epoch
            is_final = (epoch == self.epochs - 1)
                
            # Save checkpoint (only latest, best, and final)
            self.save_checkpoint(epoch + 1, total_val_loss, is_best, is_final)
        
        self.logger.info("Training completed!")
        return history
