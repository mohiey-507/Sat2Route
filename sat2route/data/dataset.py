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
            v2.RandomRotation([0, 90, 180, 270])
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
            full_image = Image.open(image_path)
            width, height = full_image.size
            mid_point = width // 2
            
            input_image = full_image.crop((0, 0, mid_point, height))
            target_image = full_image.crop((mid_point, 0, width, height))
            
            input_tensor = self.to_tensor(input_image)
            target_tensor = self.to_tensor(target_image)
            
            if self.transform:
                combined = torch.cat([input_tensor, target_tensor], dim=0)
                combined = self.transform(combined)
                
                input_channels = input_tensor.shape[0]
                input_tensor = combined[:input_channels]
                target_tensor = combined[input_channels:]
            
            input_tensor = self._resize_tensor(input_tensor)
            target_tensor = self._resize_tensor(target_tensor)
            
            return input_tensor, target_tensor
        
        except (IOError, OSError, Image.UnidentifiedImageError) as img_err:
            print(f"Error processing image {image_path}: {img_err}")
            return self.__getitem__(idx + 1)