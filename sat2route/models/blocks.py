import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
        Conv2d --> BatchNorm2d --> activation --> Dropout
    --> Conv2d --> BatchNorm2d --> activation --> Dropout
    """
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool, use_dropout: bool, activation=nn.ReLU):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout())
        layers += [nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout())
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class ContractBlock(nn.Module):
    """
    Downsampling block for UNet: ConvBlock with LeakyReLU then 2x2 max pooling.
    """
    def __init__(self, in_ch: int, use_bn: bool = True, use_dropout: bool = False):
        super().__init__()
        out_ch = in_ch * 2
        self.conv = ConvBlock(in_ch, out_ch, use_bn, use_dropout, activation=lambda inplace: nn.LeakyReLU(0.2, inplace=inplace))
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.pool(self.conv(x))

class ExpandBlock(nn.Module):
    """
    Upsampling block for UNet: ConvTranspose2d for upsample then ConvBlock with ReLU.
    """
    def __init__(self, in_ch: int, use_bn: bool = True, use_dropout: bool = False):
        super().__init__()
        mid_ch = in_ch // 2
        self.deconv = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, mid_ch, use_bn, use_dropout, activation=nn.ReLU)
    
    def forward(self, x, skip):
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
