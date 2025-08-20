import os
import sys
import torch
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat2route.losses.loss import Loss
from sat2route.models import Generator, Discriminator
from sat2route.config import get_config


class TestLoss(unittest.TestCase):
    def setUp(self):
        default_config = get_config()
        self.gen_config = default_config['model']['generator']
        self.disc_config = default_config['model']['discriminator']
        self.training_config = default_config['training']
        self.height, self.width = default_config['dataset']['target_shape']
        
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
        
        self.loss_fn = Loss(lambda_recon=self.training_config['lambda_recon'])
        
        batch_size = 2
        self.condition = torch.randn(batch_size, self.gen_config['in_channels'], self.height, self.width)
        self.real = torch.randn(batch_size, self.gen_config['out_channels'], self.height, self.width)
        
    def test_initialization(self):
        """Test if Loss initializes correctly with default parameters"""
        self.assertIsInstance(self.loss_fn, Loss)
        self.assertEqual(self.loss_fn.lambda_recon, self.training_config['lambda_recon'])
        self.assertIsInstance(self.loss_fn.adv_criterion, torch.nn.BCEWithLogitsLoss)
        self.assertIsInstance(self.loss_fn.recon_criterion, torch.nn.L1Loss)
    
    def test_generator_loss(self):
        """Test generator loss computation"""
        loss_dict = self.loss_fn(
            model=self.generator, 
            real=self.real, 
            condition=self.condition, 
            mode='generator', 
            disc=self.discriminator
        )
        
        # Check if loss dict contains expected keys
        self.assertIn('total', loss_dict)
        self.assertIn('adv', loss_dict)
        self.assertIn('recon', loss_dict)
        
        # Verify all loss values are tensors and have gradient
        for name, loss in loss_dict.items():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size([]))  # Scalar tensor
        
        # Check if total loss equals adv_loss + lambda_recon * recon_loss
        expected_total = loss_dict['adv'] + self.loss_fn.lambda_recon * loss_dict['recon']
        self.assertTrue(torch.allclose(loss_dict['total'], expected_total))
    
    def test_discriminator_loss(self):
        """Test discriminator loss computation"""
        loss_dict = self.loss_fn(
            model=self.discriminator, 
            real=self.real, 
            condition=self.condition, 
            mode='discriminator', 
            gen=self.generator
        )
        
        # Check if loss dict contains expected keys
        self.assertIn('total', loss_dict)
        self.assertIn('real', loss_dict)
        self.assertIn('fake', loss_dict)
        
        # Verify all loss values are tensors and have gradient
        for name, loss in loss_dict.items():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size([]))  # Scalar tensor
        
        # Check if total loss equals (real_loss + fake_loss) / 2
        expected_total = (loss_dict['real'] + loss_dict['fake']) / 2
        self.assertTrue(torch.allclose(loss_dict['total'], expected_total))

if __name__ == '__main__':
    unittest.main()
