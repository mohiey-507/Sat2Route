import os
import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sat2route import (
    Generator, Discriminator, Trainer, Loss,
    get_dataloaders, default_config
)

def parse_args():
    pass

def main():
    args = parse_args()
    pass

if __name__ == '__main__':
    main()