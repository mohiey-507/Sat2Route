import torch
import torch.nn as nn
from .blocks import ContractBlock, ExpandBlock

class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 32, depth: int = 6):
        super().__init__()  

        self.in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)

        # Contracting path
        ch = hidden_ch
        self.contracts = nn.ModuleList()
        for i in range(depth):
            self.contracts.append(ContractBlock(ch, use_norm=True, use_dropout=False))
            ch *= 2

        # Expanding path
        expand_ch = ch
        self.expands = nn.ModuleList()
        for i in range(depth):
            use_dropout = i < ((depth + 2) // 3)
            self.expands.append(ExpandBlock(expand_ch, use_norm=True, use_dropout=use_dropout))
            expand_ch //= 2

        self.out_conv = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)

        self._init_weights()

    def forward(self, x):
        x = self.in_conv(x)
        skips = []
        for contract in self.contracts:
            skips.append(x)
            x = contract(x)
        for expand in self.expands:
            x = expand(x, skips.pop())
        x = self.out_conv(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m is self.out_conv:
                    nn.init.xavier_normal_(m.weight)
                elif any(m in block.modules() for block in self.contracts):
                    nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
