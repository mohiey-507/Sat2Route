import torch
import torch.nn as nn
from .blocks import ContractBlock
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image translation tasks
    Classifies if image patches are real or fake rather than the entire image
    """
    def __init__(self, in_ch: int, hidden_ch: int = 32, depth: int = 4, use_dropout: bool = False, use_spectral: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        
        in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        if use_spectral:
            in_conv = spectral_norm(in_conv)
        self.in_conv = in_conv
        ch = hidden_ch
        self.contracts = nn.ModuleList()
        for _ in range(depth):
            self.contracts.append(ContractBlock(ch, use_dropout=use_dropout, use_spectral=use_spectral, activation='leaky', dropout_rate=dropout_rate))
            ch *= 2
        out_conv = nn.Conv2d(ch, 1, kernel_size=1)
        if use_spectral:
            out_conv = spectral_norm(out_conv)
        self.out_conv = out_conv
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.in_conv(x)
        for contract in self.contracts:
            x = contract(x)
        return self.out_conv(x)
