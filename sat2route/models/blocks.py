import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
        Conv2d --> InstanceNorm2d --> activation --> Dropout
    --> Conv2d --> InstanceNorm2d --> activation --> Dropout
    """
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool, use_dropout: bool, activation=nn.ReLU, norm_affine: bool = False):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=norm_affine))
        layers.append(activation(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout())
        layers += [nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=norm_affine))
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
    def __init__(self, in_ch: int, use_norm: bool = True, use_dropout: bool = False, norm_affine: bool = False):
        super().__init__()
        out_ch = in_ch * 2
        activation = lambda inplace: nn.LeakyReLU(0.2, inplace=inplace)
        self.conv = ConvBlock(in_ch, out_ch, use_norm, use_dropout, activation=activation, norm_affine=norm_affine)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
    
    def forward(self, x):
        return self.down(self.conv(x))

class ExpandBlock(nn.Module):
    """
    Upsampling block for UNet: ConvTranspose2d for upsample then ConvBlock with ReLU.
    """
    def __init__(self, in_ch: int, use_norm: bool = True, use_dropout: bool = False, norm_affine: bool = False):
        super().__init__()
        mid_ch = in_ch // 2
        self.deconv = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, mid_ch, use_norm, use_dropout, activation=nn.ReLU, norm_affine=norm_affine)
    
    def forward(self, x, skip):
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
