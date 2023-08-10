import torch
from torch import nn
from torchvision.models import vgg16
import torchvision
import cv2
import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from  layers.blocks import *

class RPN(nn.Module):
    def __init__(self, in_ch):
        super(RPN, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, in_ch, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.bbox_layer = nn.Sequential(
            ConvBlock(in_ch, 9*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.score_layer = nn.Sequential(
            ConvBlock(in_ch, 9*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.conv(x)
        bbox_proposal = self.bbox_layer(x)
        score_proposal = self.score_layer(x)

        bbox_proposal = bbox_proposal.view(batch, 8, 8, 9, 4)
        score_proposal = score_proposal.view(batch, 8, 8, 9, 2)

        return bbox_proposal, score_proposal

class MyFasterRCNN(nn.Module):
    def __init__(self, n_classes):
        super(MyFasterRCNN, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(4096, n_classes)
        self.bounding_box = nn.Linear(4096, 4)

    def forward(self, x):
        x = self.fc_layer(x)
        cls = self.classifier(x)
        bbox = self.bounding_box(x)

        return cls, bbox


if __name__ == '__main__':
    vgg = vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
    rpn = RPN(512)


    img = cv2.imread("D:\study\data\cat.jpg")
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    img /= np.array([0.485, 0.456, 0.406])

    rpn = rpn .cuda()
    vgg = vgg.cuda()
    vgg.eval()
    rpn.eval()

    img = torch.from_numpy(img).permute(2, 0, 1).cuda().unsqueeze(0).float()

    vgg_feature = vgg.forward(img)
    bbox_proposal, score_proposal = rpn(vgg_feature)

    print(bbox_proposal.shape)
    print(score_proposal.shape)
