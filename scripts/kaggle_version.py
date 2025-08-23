# %% [code]
# Clone the repository
!git clone https://github.com/mohiey-507/Sat2Route
%cd Sat2Route

# %% [code]
%%bash
echo "Download and extract dataset"
mkdir -p datasets
wget -q -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz -O datasets/maps.tar.gz
tar -zxf datasets/maps.tar.gz -C datasets/
rm datasets/maps.tar.gz

# %% [code]
# Set up Python path for imports
import os
import sys
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path.cwd()))

from sat2route import (
    Generator, Discriminator, Trainer, Loss,
    get_dataloaders, get_config
)

# %% [code] 
def get_kaggle_config():
    config = get_config()
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Visualization frequency
    display_step = 10_000
    
    return config, checkpoint_dir, display_step

# %% [code]
def train_model():
    """Set up and train the Sat2Route model in Kaggle environment"""
    
    # Get configuration for Kaggle
    config, checkpoint_dir, display_step = get_kaggle_config()
    device = torch.device(config['training']['device'])
    
    print("Setting up dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Print training info
    print("\n--- Training Configuration ---")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['dataloader']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print(f"Lambda recon: {config['training']['lambda_recon']}")
    print(f"Generator channels: {config['model']['generator']['hidden_channels']}")
    print(f"Discriminator channels: {config['model']['discriminator']['hidden_channels']}")
    print(f"Generator depth: {config['model']['generator']['depth']}")
    print(f"Discriminator depth: {config['model']['discriminator']['depth']}")
    print(f"Display step: {display_step}")
    print("----------------------------\n")
    
    # Initialize models based on config
    print("Initializing models...")
    gen_cfg = config['model']['generator']
    generator = Generator(
        in_ch=gen_cfg['in_channels'],
        out_ch=gen_cfg['out_channels'],
        hidden_ch=gen_cfg['hidden_channels'],
        depth=gen_cfg['depth']
    )
    disc_cfg = config['model']['discriminator']
    discriminator = Discriminator(
        in_ch=disc_cfg['in_channels'],
        hidden_ch=disc_cfg['hidden_channels'],
        depth=disc_cfg['depth']
    )
    
    # Initialize optimizer and loss function
    print("Setting up optimizers and loss function...")
    train_cfg = config['training']
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg['lr'],
        betas=(train_cfg['beta1'], train_cfg['beta2'])
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_cfg['lr'],
        betas=(train_cfg['beta1'], train_cfg['beta2'])
    )
    
    loss_fn = Loss(lambda_recon=config['training']['lambda_recon'])
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        loss_fn=loss_fn,
        device=device,
        log_name='kaggle_run',
        checkpoint_dir=checkpoint_dir,
        display_step=display_step,
    )
    
    # Start training
    print("\nStarting training...\n")
    trainer.fit()
    
    print("\nTraining completed successfully!")
    
    # Save final model in a Kaggle-accessible location for download
    final_model_path = os.path.join(checkpoint_dir, 'kaggle_final_model.pth')
    
    # Saving state_dict
    gen_state_dict = generator.state_dict()
    
    torch.save(
        {'generator_state_dict': gen_state_dict},
        final_model_path
    )
    
    print(f"\nFinal model saved to {final_model_path}")
    
    return generator

# %% [code]
# Run the complete training process
generator = train_model()