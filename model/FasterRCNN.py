import torch
from torch import nn
from torchvision.models import vgg16
from torch.nn import functional as F

import torchvision
from torchvision import transforms

import cv2
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from layers.blocks import *
from utils.anchor import make_anchor_box
from utils.bbox import VOC_bbox
from utils.iou import ious

class VGG16(nn.Module):
    def __init__(self, weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1):
        super(VGG16, self).__int__()
        self.vgg = vgg16(weights = weights).features

    def forward(self, x : torch.Tensor) ->  torch.Tensor:
        x = x.unsqueeze(0)
        feature = self.vgg(x)

        return feature
    

class RPN(nn.Module):
    def __init__(self, in_ch, grid_size):
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
    
    @staticmethod
    def make_anchor_box(image_size, grid_size):
        anchor_bbox_area = [128*128, 256*256, 512*512]
        anchor_bbox_ratio = [[1, 1], [2, 1], [1, 2]]
        
        img_height, img_width = image_size
        grid_height, grid_width = grid_size
        
        x_range = torch.range(0, img_width, img_width / (grid_width - 1))
        y_range = torch.range(0, img_height, img_height / (grid_height - 1))
        
        grid_x, grid_y = torch.meshgrid(x_range, y_range)
        
        xy = torch.concat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim = -1)
        xy = xy.reshape(grid_height, grid_width, 2)
        
        anchor_list = []
        
        for area in anchor_bbox_area:
            for ratio in anchor_bbox_ratio:
                w_ratio, h_ratio = ratio
                u = math.sqrt(area / (w_ratio * h_ratio))
                w, h = w_ratio * u, h_ratio * u
                wh = torch.tensor([w, h])
                wh = torch.Tensor.repeat(wh, grid_height * grid_width).view(grid_height, grid_width, 2)
                
                xywh = torch.concat([xy.clone(), wh], dim = -1).unsqueeze(2)
                anchor_list.append(xywh)
                
        anchor_boxes = torch.concat(anchor_list, dim = 2)

        return anchor_boxes

    def forward(self, x, img_size):
        x = x.unsqueeze(0)
        batch = x.shape[0]
        grid_x = x.shape[-1]
        grid_y = x.shape[-2]
        
        x = self.conv(x)
        bbox_proposal = self.bbox_layer(x)
        score_proposal = self.score_layer(x)

        bbox_proposal = bbox_proposal.view(batch, grid_y, grid_x, 9, 4)
        score_proposal = score_proposal.view(batch, grid_y, grid_x, 9, 2)

        anchor_box = self.make_anchor_box(img_size, [grid_y, grid_x])
        
        return bbox_proposal, score_proposal, anchor_box
    
    def pos_neg_sampling(self, bbox_proposal, target_bboxes, random_choice = 128):
        """
        1. Positive Sample
            1) anchor bboxes_regressor와 target bboxes의 IoU가 가장 큰것
            2) anchor bboxes regressor와 target bboxes의 IoU가 0.7 이상인 것

        2. Negative Sample
            1) anchor bboxes regressor와 target bboxe의 IoU가 0.3 이하인 것
            
        Positive sample과 Negative sample에서 128:128로 샘플링
        여기서 만약 Positive sample의 개수가 부족하다면 Negative Sample로 추가
        
        Args:
            bbox_proposal (torch.Tensor): anchor_bboxes_regressor
            target_bboxes (torch.Tensor): target box
        
        Returns:
            positive_sample (list) : sampling된 positive sample index list
            negative_sample (list) : sampling된 negative sample index list        
        """        
        positive_index = list()
        negative_index = list()
        target_index = list()
        
        bbox_proposal = bbox_proposal.clone().detach().cpu().numpy() # [cx, cy, w, h]
        target_bboxes = target_bboxes.clone().detach().cpu().numpy() /600
        
        bbox_proposal[:, :2] += bbox_proposal[:, 2:]/2 # [x, y, w, h]
        bbox_proposal[:, 2:] += bbox_proposal[:, :2] # [x1, y1, x2, y2]
        target_bboxes[:, :2] += target_bboxes[:, 2:]/2
        target_bboxes[:, 2:] += target_bboxes[:, :2]

        _ious = ious(target_bboxes ,bbox_proposal)
        
        target, positive = np.where(_ious > 0.7)
        _, negative = np.where(_ious < 0.3)
        
        target_index.extend(target)
        positive_index.extend(positive)
        negative_index.extend(negative)
        #     if positive.shape[0] ==0 :
        #         positive_index.append(i)
        #         target_index.append(np.argmax(_iou))
        #     else :
        #         positive_index.extend(positive.tolist())
        #         target_index.append(p_target)
        #     negative_index.extend(negative.tolist())
        
        negative_pedding = 0
        
        if len(positive_index) > 128:
            positive_sample_index = np.random.choice(len(positive_index), random_choice, replace = False)
        else :
            positive_sample_index = positive_index
            negative_pedding = random_choice - len(positive_index)
            
        negative_sample_index = np.random.choice(len(negative), random_choice + negative_pedding, replace = False)
        
        print(positive_sample_index)
        print(negative_sample_index)
        target_index = np.array(target_index)
        print(target_index[positive_sample_index])
        return positive_sample_index, negative_sample_index, target_index[positive_sample_index]
    
    def loss(self, 
             positive_bbox_proposal : torch.Tensor, 
             score_proposal : torch.Tensor,
             anchor_bbox : torch.Tensor, 
             target_bboxes : torch.Tensor,
             target_scores : torch.Tensor,
             positive_weight = 10)-> torch.Tensor:
                
        # score loss        
        score_loss = F.cross_entropy(score_proposal, target_scores)
        
        
        # regression loss
        scale = anchor_bbox[:, 2:].clone()
        
        positive_bbox_proposal[:, 2:] = torch.log(positive_bbox_proposal[:, 2:])
        anchor_bbox[:, 2:] = torch.log(anchor_bbox[:, 2:])
        target_bboxes[:, 2:] = torch.log(target_bboxes[:, 2:])
        
        print(positive_bbox_proposal.shape, anchor_bbox.shape)

        t_pred = positive_bbox_proposal - anchor_bbox
        t_true = target_bboxes - anchor_bbox
        
        t_pred[:, :2] /= scale
        t_true[:, :2] /= scale
        
        regression_loss = F.smooth_l1_loss(t_pred, t_true)
        
        loss = score_loss + positive_weight * regression_loss
        
        return loss 


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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    vgg = vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.cuda()
    grid_size = (25, 25)
    rpn = RPN(512, grid_size=grid_size).cuda()

    img = cv2.imread("D:\\datasets\\VOCdevkit\\VOC2012\\JPEGImages\\2007_004538.jpg")
        
    h, w, _ = img.shape
    scale = 600/min(h, w)

    target_bboxes, target_labels = VOC_bbox("D:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_004538.xml")
    target_bboxes = torch.tensor(target_bboxes).cuda()
    target_bboxes[:, 2:] -= target_bboxes[:, :2]
    target_bboxes /= scale
    
    inputs_img = np.zeros([600, 600, 3])
    img = cv2.resize(img, (int(w//scale), int(h//scale)))
    inputs_img[:int(h//scale), :int(w//scale), :] = img
    img = torch.tensor(inputs_img)
    img = img[:, :, [2, 1, 0]]/ 255.
    img = img.permute(2, 0, 1)
    
    img = transforms.Normalize(mean, std)(img).float().cuda()
    
    vgg_feature = vgg(img)
    bbox_proposal, score_proposal, anchor_box = rpn(vgg_feature, (600, 600))
    #bbox_proposal : [1, x_axis, y_axis, 9, 4]
    
    bbox_proposal = bbox_proposal.view(-1, 4)
    score_proposal = score_proposal.view(-1, 2)
    
    pos_index, neg_index, target_index = rpn.pos_neg_sampling(bbox_proposal=bbox_proposal, target_bboxes=target_bboxes)
    
    pos_bbox = bbox_proposal[pos_index]
    tar_bbox = target_bboxes[target_index]

    print(pos_bbox.shape, tar_bbox.shape)
    proposal_scores = torch.concat([score_proposal[pos_index], score_proposal[neg_index]], dim=0)

    positive_target_scores = torch.zeros([len(pos_index), 2])
    positive_target_scores[:, 0] +=1
    negative_target_scores = torch.zeros([len(target_index), 2])
    negative_target_scores[:, 1] += 1


    target_scores = torch.concat([positive_target_scores, negative_target_scores], dim = 0).cuda()
    
    print(proposal_scores.shape, target_scores.shape)

    loss = rpn.loss(pos_bbox, proposal_scores, anchor_box, tar_bbox, target_scores)
    
    print(loss)
    
    
if __name__ == '__main__':
    # anchor_boxes = RPN.make_anchor_box([600, 600], [25, 25])
    # print(anchor_boxes.shape)
    # print(anchor_boxes[0, 0, :, :])
    # print(anchor_boxes[0, 1, :, :])
    train()
    # # VGG16's feature map
    # vgg = vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).features 
    
    # # Region Proposal Network
    # grid_size = (25, 25)
    # rpn = RPN(512, grid_size=grid_size)

    # region_proposal = RegionProposal()

    # # Faster-RCNN
    # faster_rcnn = MyFasterRCNN(100)

    # # setting for test
    # rpn = rpn .cuda()
    # vgg = vgg.cuda()
    # faster_rcnn = faster_rcnn.cuda()
    # vgg = vgg.eval()
    # rpn = rpn.eval()
    # faster_rcnn = faster_rcnn.eval()

    # # PreProcessing Image
    # img_size = (800, 800)
    # img = cv2.imread("D:\study\data\cat.jpg")
    # img = cv2.resize(img, img_size)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    # img /= np.array([0.485, 0.456, 0.406])
    # img = torch.from_numpy(img).permute(2, 0, 1).cuda().unsqueeze(0).float()

    # # 1. VGG's feature extractor
    # vgg_feature = vgg.forward(img)  # [batch, 512, 25, 25]
    # print(f"vgg feature : {vgg_feature.shape}")
    # # 2. Region Proposal network, Bounding Box Regressor & Objectness Score
    # bbox_regressor, score_proposals = rpn(vgg_feature)
    # print(f"bbox_regressor : {bbox_regressor.shape}")
    # print(f"score_proposals : {score_proposals.shape}")

    # # 3. make Anchor box
    # anchor_boxes = make_anchor_box(image_size=img_size, gride_size=grid_size)
    # anchor_boxes = torch.from_numpy(anchor_boxes).cuda()
    # print(f"anchor boxes : {anchor_boxes.shape}")

    # # 4. Anchor Box & RPN's Bounding Box Regressor
    # region_proposals = region_proposal(anchor_boxes, bbox_regressor)
    # print(f"region_proposals : {region_proposals.shape}")

    # # 5. NMS
    # batch_size = region_proposals.shape[0]
    # num_of_bbox = region_proposals.shape[1] * region_proposals.shape[2] * region_proposals.shape[3]
    # index = torch.tensor([[i] * num_of_bbox for i in range(batch_size)]).flatten().cuda()
    # region_proposals = region_proposals.view(-1, 4)
    # score_proposals = score_proposals.view(-1, 2)
    # proposal_index = torchvision.ops.batched_nms(boxes = region_proposals, scores = score_proposals[:, 0], idxs=index, iou_threshold = 0.4)

    # bboxes = region_proposals[proposal_index]
    # scores = score_proposals[proposal_index]

    # print(f"nms bboxes : {bboxes.shape}")
    # print(f"nms scores : {scores.shape}")

    # # 6. top k resonproposal using socre
    # _, topk_index = torch.topk(scores[:, 0], k = 100)

    # bboxes = bboxes[topk_index]
    # scores = scores[topk_index]

    # print(f"top k bboxes : {bboxes.shape}")
    # print(f"top k scores : {scores.shape}")

    # # 7. ROI Pooling
    # scale = 25/800
    # roi_pool_feature = torchvision.ops.roi_pool(vgg_feature, [bboxes], (7, 7), scale)
    # print(f"roi_pool_feature : {roi_pool_feature.shape}")

    # # 8. Faster RCN
    # out_classes, out_bboxes = faster_rcnn(roi_pool_feature.view(roi_pool_feature.shape[0], -1))
    # print(f"out classes : {out_classes.shape}")
    # print(f"out bboxes : {out_bboxes.shape}")