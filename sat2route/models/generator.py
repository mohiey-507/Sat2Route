import torch.nn as nn
from .blocks import ContractBlock, ExpandBlock

class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 64, depth: int = 7, max_ch: int = 512):
        super().__init__()  
        assert depth < 8, "Depth should be at most 7 for UNet"

        self.in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)

        # Contracting path
        ch = hidden_ch
        skip_channels = []
        self.contracts = nn.ModuleList()
        for _ in range(depth):
            skip_channels.append(ch)
            out_ch_val = min(ch * 2, max_ch)
            self.contracts.append(ContractBlock(ch, out_ch=out_ch_val, use_norm=True, use_spectral=False, activation='leaky'))
            ch = out_ch_val

        # Expanding path
        current_ch = ch
        self.expands = nn.ModuleList()
        for i, skip_ch in enumerate(reversed(skip_channels)):
            use_dropout = i < ((depth + 2) // 3)
            self.expands.append(ExpandBlock(current_ch, skip_ch=skip_ch, use_dropout=use_dropout))
            current_ch = skip_ch

        self.out_conv = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)

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
