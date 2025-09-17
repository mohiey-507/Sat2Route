import os
import sys
import torch
import unittest
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat2route.engine.trainer import Trainer
from sat2route.models import Generator, Discriminator
from sat2route.losses.loss import Loss
from sat2route.data.data_loader import get_dataloaders
from sat2route.config import get_config

class TestTrainer(unittest.TestCase):
    def setUp(self):
        
        default_config = get_config({
                'training': {
                    'device': 'cpu',
                    'lambda_recon': 100.0,
                    'epochs': 1,
                    'lr': 0.0002,
                    'beta1': 0.5,
                    'beta2': 0.999,
                },
                'dataset': {
                'root_dir': 'sat2route/datasets/maps',
                'target_shape': (256, 256),
                'test_size': 0.2,
                'seed': 2025
                }
        })
        self.config = default_config
        self.gen_config = default_config['model']['generator']
        self.disc_config = default_config['model']['discriminator']
        self.training_config = default_config['training']
        self.dataloader_config = default_config['dataloader']
        
        # Use smaller batch size
        self.batch_size = self.dataloader_config['batch_size']
        
        # Create dataloaders
        self.train_loader, self.val_loader = get_dataloaders(default_config, batch_size=self.batch_size, test_size=0.984)
        
        # Create model instances
        self.generator = Generator(
            in_ch=self.gen_config['in_channels'],
            out_ch=self.gen_config['out_channels'],
            hidden_ch=self.gen_config['hidden_channels'],
            depth=self.gen_config['depth']
        )
        
        self.discriminator = Discriminator(
            in_ch=self.disc_config['in_channels'],
            hidden_ch=self.disc_config['hidden_channels'],
            depth=self.disc_config['depth']
        )
        
        # Create optimizer instances
        self.gen_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.training_config['lr'],
            betas=(self.training_config['beta1'], self.training_config['beta2']),
            weight_decay=1e-4
        )
        
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.training_config['lr'],
            betas=(self.training_config['beta1'], self.training_config['beta2']),
            weight_decay=1e-4
        )
        
        # Create loss function
        self.loss_fn = Loss(lambda_recon=self.training_config['lambda_recon'])
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        
        # Create trainer instance
        self.trainer = Trainer(
            config=default_config,
            generator=self.generator,
            discriminator=self.discriminator,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            loss_fn=self.loss_fn,
            checkpoint_dir=self.checkpoint_dir,
            log_name='test_trainer',
            device=torch.device('cpu')
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test if Trainer initializes correctly with default parameters"""
        self.assertIsInstance(self.trainer, Trainer)
        self.assertEqual(self.trainer.lambda_recon, self.training_config['lambda_recon'])
        self.assertEqual(self.trainer.epochs, self.training_config['epochs'])
        self.assertEqual(self.trainer.lr, self.training_config['lr'])
        self.assertEqual(self.trainer.beta1, self.training_config['beta1'])
        self.assertEqual(self.trainer.beta2, self.training_config['beta2'])
        self.assertEqual(self.trainer.device.type, 'cpu')
        self.assertIsInstance(self.trainer.generator, Generator)
        self.assertIsInstance(self.trainer.discriminator, Discriminator)
        self.assertIsInstance(self.trainer.loss_fn, Loss)
    
    def test_train_epoch(self):
        """Test a single training epoch"""
        
        losses = self.trainer.train_epoch()
        
        self.assertIn('disc', losses)
        self.assertIn('gen', losses)
        self.assertIn('total', losses['disc'])
        self.assertIn('real', losses['disc'])
        self.assertIn('fake', losses['disc'])
        self.assertIn('total', losses['gen'])
        self.assertIn('adv', losses['gen'])
        self.assertIn('recon', losses['gen'])
    
    def test_save_load_checkpoint(self):
        """Test checkpoint save and load functionality"""
        # Save a checkpoint
        val_loss = 1.234
        epoch = 1
        self.trainer.save_checkpoint(epoch, val_loss)
        
        # Verify checkpoint file exists
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Make changes to model parameters to verify they're restored
        original_params = next(self.trainer.generator.parameters()).clone()
        with torch.no_grad():
            for param in self.trainer.generator.parameters():
                param.add_(torch.ones_like(param))
        
        # Load the checkpoint
        loaded_epoch = self.trainer.load_checkpoint(checkpoint_path)
        
        # Verify epoch and params are restored
        self.assertEqual(loaded_epoch, epoch)
        self.assertEqual(self.trainer.best_val_loss, val_loss)
        loaded_params = next(self.trainer.generator.parameters())
        self.assertTrue(torch.allclose(original_params, loaded_params))

if __name__ == '__main__':
    unittest.main()
