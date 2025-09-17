import torch
import torch.nn as nn
from .blocks import ContractBlock
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image translation tasks
    Classifies if image patches are real or fake rather than the entire image
    """
    def __init__(self, in_ch: int, hidden_ch: int = 64, depth: int = 5, max_ch: int = 512,
                use_dropout: bool = False, use_spectral: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        
        in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=4, stride=2, padding=1)
        if use_spectral:
            in_conv = spectral_norm(in_conv)
        self.in_conv = in_conv

        k_down = min(depth, 2)
        ch = hidden_ch
        self.contracts = nn.ModuleList()
        for i in range(depth):
            out_ch_val = min(ch * 2, max_ch)
            down_flag = True if i < k_down else False
            self.contracts.append(
                ContractBlock(ch, out_ch=out_ch_val, use_dropout=use_dropout, use_spectral=use_spectral,
                            downsample=down_flag, activation='leaky', dropout_rate=dropout_rate
                )
            )
            ch = out_ch_val
        
        out_conv = nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral:
            out_conv = spectral_norm(out_conv)
        self.out_conv = out_conv
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.in_conv(x)
        for contract in self.contracts:
            x = contract(x)
        return self.out_conv(x)
