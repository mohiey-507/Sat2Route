import os
import sys
import torch
import unittest
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat2route.data.dataset import MapsDataset
from sat2route.data.data_loader import get_dataloaders
from sat2route.config import get_config


class TestMapsDataset(unittest.TestCase):
    def setUp(self):
        self.config = get_config()["dataset"]
        self.root_dir = self.config["root_dir"]
        self.target_shape = self.config["target_shape"]
        self.test_size = self.config["test_size"]

    def test_getitem(self):
        """Test if __getitem__ returns tensors with correct shapes"""
        train_dataset = MapsDataset(
            root_dir=self.root_dir,
            target_shape=self.target_shape,
            is_train=True,
            test_size=self.test_size
        )
        
        input_tensor, target_tensor = train_dataset[0]
        
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)
        
        self.assertEqual(input_tensor.shape[1:], torch.Size(self.target_shape))
        self.assertEqual(target_tensor.shape[1:], torch.Size(self.target_shape))
        
        self.assertTrue(torch.all(input_tensor >= 0) and torch.all(input_tensor <= 1))
        self.assertTrue(torch.all(target_tensor >= 0) and torch.all(target_tensor <= 1))


class TestDataLoader(unittest.TestCase):
    def test_get_dataloaders(self):
        """Test if dataloaders are created correctly with default parameters"""
        train_loader, val_loader = get_dataloaders()
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)

    def test_dataloader_batch(self):
        """Test if dataloader returns batches with correct shapes"""
        train_loader, _ = get_dataloaders(batch_size=4)
        
        for input_batch, target_batch in train_loader:
            self.assertEqual(input_batch.shape[0], 4)
            self.assertEqual(target_batch.shape[0], 4)
            
            self.assertEqual(input_batch.shape[1], 3) # RGB input
            self.assertEqual(target_batch.shape[1], 3) # RGB target
            
            break


if __name__ == '__main__':
    unittest.main()
