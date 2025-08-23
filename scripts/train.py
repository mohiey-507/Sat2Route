import os
import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sat2route import (
    Generator, Discriminator, Trainer, Loss,
    get_dataloaders, get_config
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Sat2Route model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--lambda-recon', type=float, default=None,
                        help='Weight of reconstruction loss (default: from config)')
    
    # Device options
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device to train on (default: auto-detected)')
    
    # Model parameters
    parser.add_argument('--g-channels', type=int, default=None,
                        help='Generator hidden channels (default: from config)')
    parser.add_argument('--d-channels', type=int, default=None,
                        help='Discriminator hidden channels (default: from config)')
    parser.add_argument('--g-depth', type=int, default=None,
                        help='Generator depth (default: from config)')
    parser.add_argument('--d-depth', type=int, default=None,
                        help='Discriminator depth (default: from config)')
    
    # Checkpoint handling
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--log-name', type=str, default='sat2route',
                        help='Name for log files (default: sat2route)')
    
    # Visualization
    parser.add_argument('--display-step', type=int, default=2000,
                        help='Frequency to display sample images during training (default: 2000)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load default config
    default_config = get_config()
    
    # Update config with command-line args if provided
    if args.lr is not None:
        default_config['training']['lr'] = args.lr
    if args.lambda_recon is not None:
        default_config['training']['lambda_recon'] = args.lambda_recon
    if args.epochs is not None:
        default_config['training']['epochs'] = args.epochs
    if args.g_channels is not None:
        default_config['model']['generator']['hidden_channels'] = args.g_channels
    if args.d_channels is not None:
        default_config['model']['discriminator']['hidden_channels'] = args.d_channels
    if args.g_depth is not None:
        default_config['model']['generator']['depth'] = args.g_depth
    if args.d_depth is not None:
        default_config['model']['discriminator']['depth'] = args.d_depth
    if args.batch_size is not None:
        default_config['dataloader']['batch_size'] = args.batch_size
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )
    
    train_loader, val_loader = get_dataloaders(default_config)
    
    # Print training info
    print(f"Training with:")
    print(f"  - Device: {device}")
    print(f"  - Epochs: {default_config['training']['epochs']}")
    print(f"  - Batch size: {default_config['dataloader']['batch_size']}")
    print(f"  - Learning rate: {default_config['training']['lr']}")
    print(f"  - Lambda recon: {default_config['training']['lambda_recon']}")
    print(f"  - Generator channels: {default_config['model']['generator']['hidden_channels']}")
    print(f"  - Discriminator channels: {default_config['model']['discriminator']['hidden_channels']}")
    
    # Initialize models based on config
    generator = Generator(
        in_ch=default_config['model']['generator']['in_channels'],
        out_ch=default_config['model']['generator']['out_channels'],
        hidden_ch=default_config['model']['generator']['hidden_channels'],
        depth=default_config['model']['generator']['depth']
    )
    
    discriminator = Discriminator(
        in_ch=default_config['model']['discriminator']['in_channels'],
        hidden_ch=default_config['model']['discriminator']['hidden_channels'],
        depth=default_config['model']['discriminator']['depth']
    )
    
    # Initialize optimizer and loss function
    gen_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=default_config['training']['lr'],
        betas=(default_config['training']['beta1'], default_config['training']['beta2'])
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=default_config['training']['lr'],
        betas=(default_config['training']['beta1'], default_config['training']['beta2'])
    )
    
    loss_fn = Loss(lambda_recon=default_config['training']['lambda_recon'])
    
    # Initialize trainer
    trainer = Trainer(
        config=default_config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        loss_fn=loss_fn,
        device=device,
        log_name=args.log_name,
        checkpoint_dir=args.checkpoint_dir,
        display_step=args.display_step,
    )
    
    # Start training
    trainer.fit(resume_from=args.resume)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()