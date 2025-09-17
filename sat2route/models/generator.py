import torch
import torch.nn as nn
from .blocks import ContractBlock

class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 64, depth: int = 8, max_ch: int = 512, input_spatial: int = 256):
        super().__init__()  
        self.in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        
        # Contracting path
        ch = hidden_ch
        skip_channels = []
        self.contracts = nn.ModuleList()
        for i in range(depth):
            skip_channels.append(ch)
            out_ch_val = min(ch * 2, max_ch)
            spatial_after = input_spatial // (2 ** (i + 1))
            use_norm_flag = True if spatial_after > 1 else False
            self.contracts.append(
                ContractBlock(
                    ch, out_ch=out_ch_val, use_norm=use_norm_flag,
                    use_spectral=False, activation='leaky'
                )
            )
            ch = out_ch_val

        # Expanding path
        current_ch = ch
        self.decoders = nn.ModuleList()
        for i, skip_ch in enumerate(reversed(skip_channels)):
            out_ch_decoder = skip_ch
            use_dropout = i < ((depth + 2) // 3)
            decoder_layers = []
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.ConvTranspose2d(current_ch, out_ch_decoder, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.InstanceNorm2d(out_ch_decoder, affine=True))
            if use_dropout:
                decoder_layers.append(nn.Dropout2d(0.5))
            self.decoders.append(nn.Sequential(*decoder_layers))
            current_ch = out_ch_decoder * 2

        self.out_conv = nn.Conv2d(current_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        skips = []
        for contract in self.contracts:
            skips.append(x)
            x = contract(x)

        for decoder in self.decoders:
            x = decoder(x)           # (N, out_ch, H, W)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)  # (N, 2*out_ch, H, W) -> next decoder input
        x = self.out_conv(x)
        return x
