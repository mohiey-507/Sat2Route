import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
class ConvBlock(nn.Module):
    """
        Conv2d --> InstanceNorm2d --> activation --> Dropout
    --> Conv2d --> InstanceNorm2d --> activation --> Dropout
    """
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool, use_dropout: bool,
                downsample: bool, use_spectral: bool, activation: str, dropout_rate: float):
        super().__init__()
        layers = []
        use_norm = use_norm and not use_spectral
        bias = not use_norm 
        stride = 2 if downsample else 1
        act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        if use_spectral:
            conv1 = spectral_norm(conv1)
        layers.append(conv1)
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(act)
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))

        conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)
        if use_spectral:
            conv2 = spectral_norm(conv2)
        layers.append(conv2)
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(act)
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class ContractBlock(nn.Module):
    """
    Downsampling block for UNet: ConvBlock with LeakyReLU.
    """
    def __init__(self, in_ch: int, use_norm: bool = False, use_dropout: bool = False,
                use_spectral: bool = True, activation='relu', dropout_rate: float = 0.5):
        super().__init__()
        out_ch = in_ch * 2
        self.conv = ConvBlock(in_ch, out_ch, use_norm=use_norm, use_dropout=use_dropout, downsample=True,
                            activation=activation, use_spectral=use_spectral, dropout_rate=dropout_rate)
    
    def forward(self, x):
        return self.conv(x)

class ExpandBlock(nn.Module):
    """
    Upsampling block for UNet: ConvTranspose2d for upsample then ConvBlock with ReLU.
    """
    def __init__(self, in_ch: int, use_dropout: bool = False,
                dropout_rate: float = 0.5):
        super().__init__()
        mid_ch = in_ch // 2
        self.deconv = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, mid_ch, use_norm=True, use_dropout=use_dropout, downsample=False, use_spectral=False, activation='relu', dropout_rate=dropout_rate)

    def forward(self, x, skip):
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
