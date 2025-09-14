import os
import glob
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Optional

class MapsDataset(Dataset):
    def __init__(self, root_dir: str, target_shape: tuple[int, int], is_train: bool, test_size: float, seed: Optional[int] = None):
        super().__init__()
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.is_train = is_train
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        train_images = glob.glob(os.path.join(root_dir, 'train', '*.jpg'))
        val_images = glob.glob(os.path.join(root_dir, 'val', '*.jpg'))
        all_images = train_images + val_images
        
        split_idx = int(len(all_images) * (1 - test_size))
        self.image_paths = all_images[:split_idx] if is_train else all_images[split_idx:]
        
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _apply_synchronized_transforms(
            self, input_pil: Image.Image, target_pil: Image.Image
        ) -> tuple[Image.Image, Image.Image]:
        """
        Apply synchronized augmentations to input and target PIL images.
        """
        if not self.is_train:
            input_pil = TF.resize(input_pil, self.target_shape, interpolation=Image.BILINEAR)
            target_pil = TF.resize(target_pil, self.target_shape, interpolation=Image.NEAREST)
            return input_pil, target_pil

        # Random horizontal flip
        if random.random() < 0.5:
            input_pil = TF.hflip(input_pil)
            target_pil = TF.hflip(target_pil)

        # Random vertical flip
        if random.random() < 0.5:
            input_pil = TF.vflip(input_pil)
            target_pil = TF.vflip(target_pil)

        # RandomResizedCrop
        i, j, h, w = v2.RandomResizedCrop.get_params(input_pil, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        input_pil = TF.resized_crop(input_pil, i, j, h, w, size=self.target_shape, interpolation=Image.BILINEAR)
        target_pil = TF.resized_crop(target_pil, i, j, h, w, size=self.target_shape, interpolation=Image.NEAREST)

        return input_pil, target_pil
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.image_paths):
            idx = idx % len(self.image_paths)
        image_path = self.image_paths[idx]

        try:
            full_image = Image.open(image_path).convert('RGB')
            width, height = full_image.size
            mid_point = width // 2
            
            input_image = full_image.crop((0, 0, mid_point, height))
            target_image = full_image.crop((mid_point, 0, width, height))

            input_image, target_image = self._apply_synchronized_transforms(input_image, target_image)
            
            input_tensor = self.to_tensor(input_image)
            target_tensor = self.to_tensor(target_image)
            
            return input_tensor, target_tensor
        
        except (IOError, OSError, Image.UnidentifiedImageError) as img_err:
            print(f"Error processing image {image_path}: {img_err}")
            return self.__getitem__(idx + 1)