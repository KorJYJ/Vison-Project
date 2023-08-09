import torch
import sys
import os
sys.path.append(os.path.dirname(__path__))

from layers.blocks import *


class MyResnet(torch.nn.Module):
    def __init__(self,):
        super(MyResnet, self).__init__()
        self.blocks = torch.nn.ModuleList([
            ConvBlock(3, 64, 7, 2, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride = 2),

            ResBottleBlock(64, 256, skip_conv = True),
            ResBottleBlock(256, 256, skip_conv=False),
            ResBottleBlock(256, 256, skip_conv=False),

            ResBottleBlock(256, 512, kernel_size=(2, 2), stride = 2, skip_conv=True),
            ResBottleBlock(512, 512, skip_conv=False),
            ResBottleBlock(512, 512, skip_conv=False),
            ResBottleBlock(512, 512, skip_conv=False),

            ResBottleBlock(512, 1024, kernel_size=(2, 2), stride = 2, skip_conv=True),
            ResBottleBlock(1024, 1024, skip_conv=False),
            ResBottleBlock(1024, 1024, skip_conv=False),
            ResBottleBlock(1024, 1024, skip_conv=False),
            ResBottleBlock(1024, 1024, skip_conv=False),
            ResBottleBlock(1024, 1024, skip_conv=False),

            ResBottleBlock(1024, 2048, kernel_size=(2, 2), stride = 2, skip_conv=True),
            ResBottleBlock(2048, 2048, skip_conv=False),
            ResBottleBlock(2048, 2048, skip_conv=False),
            torch.nn.AvgPool2d(7),

            classifier()
        ])

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
            
        return x