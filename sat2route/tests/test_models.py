import os 
import sys 
import torch 
import unittest 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat2route.models import Generator, Discriminator
from sat2route.config import get_config

class TestUNetGenerator(unittest.TestCase):
    def setUp(self):
        default_config = get_config()
        self.gen_config = default_config['model']['generator']
        self.dataset_config = default_config['dataset']
        self.in_channels = self.gen_config['in_channels']
        self.out_channels = self.gen_config['out_channels']
        self.hidden_channels = self.gen_config['hidden_channels']
        self.depth = self.gen_config['depth']
        self.height, self.width = self.dataset_config['target_shape']
        
        self.model = Generator(
            in_ch=self.in_channels,
            out_ch=self.out_channels,
            hidden_ch=self.hidden_channels,
            depth=self.depth
        )
        
    def test_initialization(self):
        """Test if UNet initializes correctly with default parameters"""
        self.assertIsInstance(self.model, Generator)
        self.assertEqual(len(self.model.contracts), self.depth)
        self.assertEqual(len(self.model.decoders), self.depth)
        self.assertEqual(self.model.out_conv.out_channels, self.out_channels)
    
    def test_forward_pass(self):
        """Test if forward pass produces correct output shape"""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.in_channels, self.height, self.width)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, self.out_channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)



class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        default_config = get_config()
        self.disc_config = default_config['model']['discriminator']
        self.dataset_config = default_config['dataset']
        self.in_channels = self.disc_config['in_channels']
        self.hidden_channels = self.disc_config['hidden_channels']
        self.depth = self.disc_config['depth']
        self.height, self.width = self.dataset_config['target_shape']
        
        self.model = Discriminator(
            in_ch=self.in_channels,
            hidden_ch=self.hidden_channels,
            depth=self.depth
        )
    
    def test_initialization(self):
        """Test if Discriminator initializes correctly with default parameters"""
        self.assertIsInstance(self.model, Discriminator)
        self.assertEqual(len(self.model.contracts), self.depth)
        self.assertEqual(self.model.out_conv.out_channels, 1)
    
    def test_forward_pass(self):
        """Test if forward pass produces correct output shape"""
        batch_size = 2
        x_channels = self.in_channels // 2
        y_channels = self.in_channels // 2
        
        x_tensor = torch.randn(batch_size, x_channels, self.height, self.width)
        y_tensor = torch.randn(batch_size, y_channels, self.height, self.width)
        
        with torch.no_grad():
            output = self.model(x_tensor, y_tensor)
        
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)
        self.assertLess(output.shape[2], self.height)
        self.assertLess(output.shape[3], self.width)

if __name__ == '__main__':
    unittest.main()
