import torch.nn as nn
from torch.nn.utils import spectral_norm
class ConvBlock(nn.Module):
    """
        Conv2d(k=4, stride={2|1}, padding=1) -> InstanceNorm2d (optional) -> Act -> Dropout (optional)
    """
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool, use_dropout: bool,
                downsample: bool, use_spectral: bool, activation: str, dropout_rate: float):
        super().__init__()
        layers = []
        use_norm = use_norm and not use_spectral
        bias = not use_norm 
        stride = 2 if downsample else 1
        act = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)

        conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=bias)
        if use_spectral:
            conv = spectral_norm(conv)
        layers.append(conv)
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(act)
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class ContractBlock(nn.Module):
    """
    Downsampling block: ConvBlock with LeakyReLU.
    """
    def __init__(self, in_ch: int, out_ch: int = None, use_norm: bool = False, use_dropout: bool = False,
                use_spectral: bool = True, downsample: bool = True, activation='relu', dropout_rate: float = 0.5):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch * 2
        self.conv = ConvBlock(in_ch, out_ch, use_norm=use_norm, use_dropout=use_dropout, downsample=downsample,
                            use_spectral=use_spectral, activation=activation, dropout_rate=dropout_rate)
    
    def forward(self, x):
        return self.conv(x)
