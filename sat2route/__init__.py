from .models.generator import UNet as Generator
from .models.discriminator import Discriminator

__all__ = [
    'Generator',
    'Discriminator',
]