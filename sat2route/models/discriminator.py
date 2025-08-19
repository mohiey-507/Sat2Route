import torch
import torch.nn as nn
from .blocks import ContractBlock

class Discriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image translation tasks
    Classifies if image patches are real or fake rather than the entire image
    """
    def __init__(self, in_ch: int, hidden_ch: int = 32, depth: int = 4, norm_affine: bool = False):
        super().__init__()
        
        self.in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        ch = hidden_ch
        self.contracts = nn.ModuleList()
        for _ in range(depth):
            self.contracts.append(ContractBlock(ch, use_norm=True, use_dropout=False, norm_affine=norm_affine))
            ch *= 2
        self.out_conv = nn.Conv2d(ch, 1, kernel_size=1)

        self._init_weights()
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.in_conv(x)
        for contract in self.contracts:
            x = contract(x)
        return self.out_conv(x)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
