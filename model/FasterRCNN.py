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

from layers.blocks import *
from utils.anchor import make_anchor_box
from utils.bbox import VOC_bbox

class VGG16(nn.Module):
    def __init__(self, weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1):
        super(VGG16, self).__int__()
        self.vgg = vgg16(weights = weights).features


    def forward(self, x):
        feature = self.vgg(x)

        return feature
    

class RPN(nn.Module):
    def __init__(self, in_ch, grid_size):
        super(RPN, self).__init__()

        self.grid_x = grid_size[1]
        self.grid_y = grid_size[0]

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

        bbox_proposal = bbox_proposal.view(batch, self.grid_y, self.grid_x, 9, 4)
        score_proposal = score_proposal.view(batch, self.grid_y, self.grid_x, 9, 2)

        return bbox_proposal, score_proposal


class RegionProposal(nn.Module):
    def __init__(self):
        super(RegionProposal, self).__init__()

    def forward(self, anchor, bbox_regressor):
        bboxes = anchor * bbox_regressor
        bbxoes = bboxes.float()
        return bbxoes

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

def train():
    vgg = vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.cuda()
    rpn = RPN(512, grid_size=grid_size).cuda()

    img = cv2.imread("d:\\datasets\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000925.jpg")
    bboxes, labels = VOC_bbox("d:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_000925.xml")


if __name__ == '__main__':
    # VGG16's feature map
    vgg = vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).features 
    
    # Region Proposal Network
    grid_size = (25, 25)
    rpn = RPN(512, grid_size=grid_size)

    region_proposal = RegionProposal()

    # Faster-RCNN
    faster_rcnn = MyFasterRCNN(100)

    # setting for test
    rpn = rpn .cuda()
    vgg = vgg.cuda()
    faster_rcnn = faster_rcnn.cuda()
    vgg = vgg.eval()
    rpn = rpn.eval()
    faster_rcnn = faster_rcnn.eval()

    # PreProcessing Image
    img_size = (800, 800)
    img = cv2.imread("D:\study\data\cat.jpg")
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    img /= np.array([0.485, 0.456, 0.406])
    img = torch.from_numpy(img).permute(2, 0, 1).cuda().unsqueeze(0).float()

    # 1. VGG's feature extractor
    vgg_feature = vgg.forward(img)  # [batch, 512, 25, 25]
    print(f"vgg feature : {vgg_feature.shape}")
    # 2. Region Proposal network, Bounding Box Regressor & Objectness Score
    bbox_regressor, score_proposals = rpn(vgg_feature)
    print(f"bbox_regressor : {bbox_regressor.shape}")
    print(f"score_proposals : {score_proposals.shape}")

    # 3. make Anchor box
    anchor_boxes = make_anchor_box(image_size=img_size, gride_size=grid_size)
    anchor_boxes = torch.from_numpy(anchor_boxes).cuda()
    print(f"anchor boxes : {anchor_boxes.shape}")

    # 4. Anchor Box & RPN's Bounding Box Regressor
    region_proposals = region_proposal(anchor_boxes, bbox_regressor)
    print(f"region_proposals : {region_proposals.shape}")

    # 5. NMS
    batch_size = region_proposals.shape[0]
    num_of_bbox = region_proposals.shape[1] * region_proposals.shape[2] * region_proposals.shape[3]
    index = torch.tensor([[i] * num_of_bbox for i in range(batch_size)]).flatten().cuda()
    region_proposals = region_proposals.view(-1, 4)
    score_proposals = score_proposals.view(-1, 2)
    proposal_index = torchvision.ops.batched_nms(boxes = region_proposals, scores = score_proposals[:, 0], idxs=index, iou_threshold = 0.4)

    bboxes = region_proposals[proposal_index]
    scores = score_proposals[proposal_index]

    print(f"nms bboxes : {bboxes.shape}")
    print(f"nms scores : {scores.shape}")

    # 6. top k resonproposal using socre
    _, topk_index = torch.topk(scores[:, 0], k = 100)

    bboxes = bboxes[topk_index]
    scores = scores[topk_index]

    print(f"top k bboxes : {bboxes.shape}")
    print(f"top k scores : {scores.shape}")

    # 7. ROI Pooling
    scale = 25/800
    roi_pool_feature = torchvision.ops.roi_pool(vgg_feature, [bboxes], (7, 7), scale)
    print(f"roi_pool_feature : {roi_pool_feature.shape}")

    # 8. Faster RCN
    out_classes, out_bboxes = faster_rcnn(roi_pool_feature.view(roi_pool_feature.shape[0], -1))
    print(f"out classes : {out_classes.shape}")
    print(f"out bboxes : {out_bboxes.shape}")