import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from PIL import Image

class MapsDataset(Dataset):
    def __init__(self, root_dir:str, target_shape:tuple, is_train:bool, test_size:float):
        super().__init__()
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.is_train = is_train
        
        # Collect and sort all image paths
        train_images = glob.glob(os.path.join(root_dir, 'train', '*.jpg'))
        val_images = glob.glob(os.path.join(root_dir, 'val', '*.jpg'))
        all_images = train_images + val_images
        
        split_idx = int(len(all_images) * (1 - test_size))
        self.image_paths = all_images[:split_idx] if is_train else all_images[split_idx:]
        
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        # Training transforms
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=10),
            v2.RandomResizedCrop(self.target_shape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            v2.RandomAffine(degrees=0,translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-5, 5, -5, 5)),
            v2.RandomPerspective(distortion_scale=0.3, p=0.3),
            v2.RandomCrop(size=self.target_shape, padding=10),
        ]) if is_train else None

    def __len__(self):
        return len(self.image_paths)
    
    def _resize_tensor(self, tensor):
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.target_shape, mode='bilinear', align_corners=False)
        return tensor.squeeze(0)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            idx = idx % len(self.image_paths)
        image_path = self.image_paths[idx]

        try:
            full_image = Image.open(image_path).convert('RGB')

            if self.transform:
                full_image = self.transform(full_image)

            width, height = full_image.size
            mid_point = width // 2
            
            input_image = full_image.crop((0, 0, mid_point, height))
            target_image = full_image.crop((mid_point, 0, width, height))
            
            input_tensor = self.to_tensor(input_image)
            target_tensor = self.to_tensor(target_image)
            
            input_tensor = self._resize_tensor(input_tensor)
            target_tensor = self._resize_tensor(target_tensor)
            
            return input_tensor, target_tensor
        
        except (IOError, OSError, Image.UnidentifiedImageError) as img_err:
            print(f"Error processing image {image_path}: {img_err}")
            return self.__getitem__(idx + 1)