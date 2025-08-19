# %% [code]
# Clone the repository
!git clone -b amp https://github.com/mohiey-507/Sat2Route
%cd Sat2Route

# %% [code]
# Download and extract dataset
!mkdir -p datasets
!wget -q -N http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz -O datasets/maps.tar.gz
!tar -zxf datasets/maps.tar.gz -C datasets/
!rm datasets/maps.tar.gz

# %% [code]
# Set up Python path for imports
import os
import sys
import torch
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Add project to path
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from sat2route import (
    Generator, Discriminator, Trainer, Loss,
    get_dataloaders, default_config
)

# Update the dataset path in the config to point to the correct location
print("Updating dataset configuration for Kaggle environment...")
default_config['dataset']['root_dir'] = 'sat2route/datasets/maps'
print(f"Dataset root directory set to: {default_config['dataset']['root_dir']}")

print("Setting up training configuration...")

# %% [code] 
# Parse arguments with default values for Kaggle
def get_kaggle_config():
    # Training parameters
    epochs = 100
    batch_size = 32
    lr = 2e-4
    lambda_recon = 180
    
    # Device configuration - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for multiple GPUs
    ngpu = torch.cuda.device_count()
    if device.type == 'cuda' and ngpu > 1:
        print(f"Using {ngpu} GPUs!")
        # Scale batch size by the number of GPUs
        batch_size *= ngpu
    else:
        print(f"Using device: {device}")
    
    # Model parameters
    g_hidden_channels = 32
    d_hidden_channels = 8
    g_depth = 6
    d_depth = 4
    
    # Checkpoint handling
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Visualization frequency
    display_step = 50
    
    # Update config
    config = default_config
    config['training']['epochs'] = epochs
    config['training']['lr'] = lr
    config['training']['lambda_recon'] = lambda_recon
    config['dataloader']['batch_size'] = batch_size
    config['model']['generator']['hidden_channels'] = g_hidden_channels
    config['model']['generator']['depth'] = g_depth
    config['model']['discriminator']['hidden_channels'] = d_hidden_channels
    config['model']['discriminator']['depth'] = d_depth
    
    return config, device, checkpoint_dir, display_step

# %% [code]
def train_model():
    """Set up and train the Sat2Route model in Kaggle environment"""
    
    # Get configuration for Kaggle
    config, device, checkpoint_dir, display_step = get_kaggle_config()
    
    print("Setting up dataloaders...")
    train_loader, val_loader = get_dataloaders()
    
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
    generator = Generator(
        in_ch=config['model']['generator']['in_channels'],
        out_ch=config['model']['generator']['out_channels'],
        hidden_ch=config['model']['generator']['hidden_channels'],
        depth=config['model']['generator']['depth']
    )
    
    discriminator = Discriminator(
        in_ch=config['model']['discriminator']['in_channels'],
        hidden_ch=config['model']['discriminator']['hidden_channels'],
        depth=config['model']['discriminator']['depth']
    )
    
    # Apply DataParallel if multiple GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print("Applying torch.nn.DataParallel to models...")
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
    
    # Initialize optimizer and loss function
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    loss_fn = Loss(lambda_recon=config['training']['lambda_recon'])
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = Trainer(
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
    
    # Handle DataParallel wrapper when saving state_dict
    gen_state_dict = generator.module.state_dict() if isinstance(generator, torch.nn.DataParallel) else generator.state_dict()
    disc_state_dict = discriminator.module.state_dict() if isinstance(discriminator, torch.nn.DataParallel) else discriminator.state_dict()
    
    torch.save({
        'generator_state_dict': gen_state_dict,
        'discriminator_state_dict': disc_state_dict,
        'config': config
    }, final_model_path)
    
    print(f"\nFinal model saved to {final_model_path}")
    
    # Generate and save some sample images
    print("\nGenerating sample predictions...")
    generate_samples(generator, val_loader, device)
    
    return generator, discriminator

# %% [code]
def generate_samples(generator, val_loader, device, num_samples=5):
    """Generate and display sample predictions"""
    generator.eval()
    
    # Create a figure to display samples
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    fig.suptitle('Sample Predictions', fontsize=16)
    
    # Set column titles
    axs[0, 0].set_title('Satellite Image (Input)')
    axs[0, 1].set_title('Ground Truth Map')
    axs[0, 2].set_title('Generated Map')
    
    # Get samples from validation set
    iterator = iter(val_loader)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get a batch
            satellite_imgs, map_imgs = next(iterator)
            
            # Move to device
            satellite_imgs = satellite_imgs.to(device)
            map_imgs = map_imgs.to(device)
            
            # Generate prediction
            fake_maps = generator(satellite_imgs)
            
            # Get the first image from the batch
            satellite_img = satellite_imgs[0].cpu().permute(1, 2, 0)
            real_map = map_imgs[0].cpu().permute(1, 2, 0)
            fake_map = fake_maps[0].cpu().permute(1, 2, 0)
            
            # Plot
            axs[i, 0].imshow(satellite_img)
            axs[i, 1].imshow(real_map.squeeze())
            axs[i, 2].imshow(fake_map.squeeze())
            
            # Remove axis ticks
            for j in range(3):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

# %% [code]
# Run the complete training process
generator, discriminator = train_model()