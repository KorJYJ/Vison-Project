import torch
from torch import nn
from torch.nn import functional as F

class ResBottleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, stride = 1, skip_conv = False):
        super(ResBottleBlock, self).__init__()

        self.blocks = nn.ModuleList([
            ConvBlock(in_ch, in_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(),
            ConvBlock(in_ch, in_ch, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            ConvBlock(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        ])

        self.skip_conv = skip_conv
        if skip_conv:
            self.skip_connetion_block = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride = stride)


    def forward(self, x):
        f_x = x.clone()
        for layer in self.blocks:
            f_x = layer(f_x)
        
        if self.skip_conv:
            x = self.skip_connetion_block(x)
        
        h_x = F.relu(f_x + x)
        
        return h_x
    

class ConvBlock(nn.ModuleList):
    def __init__(self, in_ch, out_ch, kernel_size, stride,  padding):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.layers(x)
        return out
    

class classifier(nn.Module):
    def __init__(self, ):
        super(classifier, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ])

    def forward(self, x):
        x = x.squeeze()
        for layer in self.blocks:
            x = layer(x)

        return x