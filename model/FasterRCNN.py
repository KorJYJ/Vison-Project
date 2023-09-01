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
    
    @staticmethod
    def make_anchor_box(image_size, grid_size):
        "return x1, y1, x2, y2"
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
        anchor_boxes[..., 2:] += anchor_boxes[..., :2]
        return anchor_boxes
    
    
    def pos_neg_sampling(self, anchor, target, random_choice = 128):
        anchor = anchor.clone().cpu().numpy()
        target = target.clone().cpu().numpy()
        
        # 이미지 밖으로 box가 나자기 않도록 만들기

        _ious = ious(target, anchor)

        pos_target, pos_anc = np.where(_ious > 0.7)
        _, neg_anc = np.where(_ious < 0.3)

        if len(pos_anc) > 128:
            positive_sample_index = np.random.choice(range(len(pos_anc)), random_choice, replace = False)
        else :
            positive_sample_index = np.arange(len(pos_anc))
            negative_pedding = random_choice - len(pos_anc)
            
        negative_sample_index = np.random.choice(range(len(neg_anc)), random_choice + negative_pedding, replace = False)

        return pos_target[positive_sample_index], pos_anc[positive_sample_index], neg_anc[negative_sample_index]

    
    def forward(self, x, img_size, target = None):
        x = x.unsqueeze(0)
        batch = x.shape[0]
        grid_x = x.shape[-1]
        grid_y = x.shape[-2]
        
        x = self.conv(x)
        bbox_regressor = self.bbox_layer(x) # dx, dy, dw, dh || [gird_y, grid_x, 9 * 4]
        score_proposal = self.score_layer(x) # [gird_y, grid_x, 9 * 2]

        bbox_regressor = bbox_regressor.view(batch, grid_y, grid_x, 9, 4)
        score_proposal = score_proposal.view(batch, grid_y, grid_x, 9, 2)
        anchor_box = self.make_anchor_box(img_size, [grid_y, grid_x]).cuda()

        loss = None

        if target is not None:
            anchor_box = anchor_box.view(-1, 4)
            target = target.view(-1, 4)
            bbox_regressor = bbox_regressor.view(-1, 4)
            score_proposal = score_proposal.view(-1, 2)

            pos_taget_idx, pos_anc_idx, neg_anc_idx = self.pos_neg_sampling(anchor_box, target)
            
            pos_neg_idx = np.concatenate([pos_anc_idx, neg_anc_idx])

            bbox_regressor = bbox_regressor[pos_anc_idx]
            score_proposal = score_proposal[pos_neg_idx]
            pos_anchor_box = anchor_box[pos_anc_idx]
            target = target[pos_taget_idx]
            target_scores = torch.zeros_like(score_proposal)
            target_scores[:len(pos_anc_idx), 0] = 1
            target_scores[len(pos_anc_idx):, 1] = 1

            target_regressor = self.box_encode(pos_anchor_box, target)

            loss = self.compute_loss(bbox_regressor, score_proposal, target_regressor, target_scores)

        return loss

    def box_encode(self, anchor, bbox):
        anchor[:, 2:] -= anchor[:, :2] # [x, y, w, h]
        bbox[:, 2:] -= bbox[:, :2] # [x, y, w, h]

        anchor[:, :2] += anchor[:, 2:]/2 # [cx, cy, w, h]
        bbox[:, :2] += bbox[:, 2:]/2 # [cx, cy, w, h]

        dx = (bbox[:, 0:1]- anchor[:, 0:1]) / anchor[:, 2:3]
        dy = (bbox[:, 1:2]- anchor[:, 1:2]) / anchor[:, 3:4]
        dw = torch.log(bbox[:, 2:3]/anchor[:, 2:3])
        dh = torch.log(bbox[:, 3:4]/anchor[:, 3:4])
        target_regressor = torch.concat([dx, dy, dw, dh], dim = 1)
        return target_regressor

    def compute_loss(self, 
             bbox_regressor : torch.Tensor, 
             scores : torch.Tensor,
             target_regressor : torch.Tensor, 
             target_scores : torch.Tensor,
             positive_weight = 10)-> torch.Tensor:
                
        # score loss        
        score_loss = F.cross_entropy(scores, target_scores)        
        
        # regression loss
        regression_loss = F.smooth_l1_loss(bbox_regressor, target_regressor)
        
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
    rpn = RPN(512).cuda()

    faster_rcnn = nn.ModuleList([vgg, rpn])

    img = cv2.imread("D:\\datasets\\VOCdevkit\\VOC2012\\JPEGImages\\2007_004538.jpg")
        
    h, w, _ = img.shape
    scale = 600/min(h, w)

    target_bboxes, target_labels = VOC_bbox("D:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_004538.xml")
    target_bboxes = torch.tensor(target_bboxes).cuda()
    # target_bboxes[:, 2:] -= target_bboxes[:, :2]
    target_bboxes *= scale
    
    
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    h, w, _ = img.shape
    img = torch.tensor(img)
    img = img[:, :, [2, 1, 0]]/ 255.
    img = img.permute(2, 0, 1)
    
    img = transforms.Normalize(mean, std)(img).float().cuda()

    optimizer  = torch.optim.SGD(faster_rcnn.parameters(), lr = 1e-4, momentum=0.9)
    

    for epoch in range(50):
        vgg_feature = vgg(img)
        loss = rpn(vgg_feature, (h, w), target_bboxes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"{epoch} : loss={loss}")
    #bbox_proposal : [1, x_axis, y_axis, 9, 4]
    
    # bbox_proposal = bbox_proposal.view(-1, 4)
    # score_proposal = score_proposal.view(-1, 2)
    
    # pos_index, neg_index, target_index = rpn.pos_neg_sampling(bbox_proposal=bbox_proposal, target_bboxes=target_bboxes)
    
    # pos_bbox = bbox_proposal[pos_index]
    # tar_bbox = target_bboxes[target_index]

    # print(pos_bbox.shape, tar_bbox.shape)
    # proposal_scores = torch.concat([score_proposal[pos_index], score_proposal[neg_index]], dim=0)

    # positive_target_scores = torch.zeros([len(pos_index), 2])
    # positive_target_scores[:, 0] +=1
    # negative_target_scores = torch.zeros([len(target_index), 2])
    # negative_target_scores[:, 1] += 1


    # target_scores = torch.concat([positive_target_scores, negative_target_scores], dim = 0).cuda()
    
    # print(proposal_scores.shape, target_scores.shape)

    # loss = rpn.loss(pos_bbox, proposal_scores, anchor_box, tar_bbox, target_scores)
    
    # print(loss)
    
    
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