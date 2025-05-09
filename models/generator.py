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

class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 32, depth: int = 6):
        super().__init__()  

        self.in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)

        # Contracting path
        ch = hidden_ch
        self.contracts = nn.ModuleList()
        for i in range(depth):
            self.contracts.append(ContractBlock(ch, use_bn=True, use_dropout=False))
            ch *= 2

        # Expanding path
        expand_ch = ch
        self.expands = nn.ModuleList()
        for i in range(depth):
            use_dropout = i < ((depth + 2) // 3)
            self.expands.append(ExpandBlock(expand_ch, use_bn=True, use_dropout=use_dropout))
            expand_ch //= 2

        self.out_conv = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(x)

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

if __name__ == "__main__":
    model = UNet(3, 3)
    print(model)

    total_params_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    print(f"Model size: {total_params_mb:.2f} MB")