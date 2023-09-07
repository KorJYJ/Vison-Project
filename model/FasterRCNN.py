import copy
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

from typing import Union, List, Tuple

class VGG16(nn.Module):
    def __init__(self, weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1):
        super(VGG16, self).__int__()
        self.vgg = vgg16(weights = weights).features

    def forward(self, x : torch.Tensor) ->  torch.Tensor:
        feature = self.vgg(x)

        return feature
    

class RPN(nn.Module):
    def __init__(self, in_ch : int):
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
    def make_anchor_box(image_size,
                        grid_size):
        "return x1, y1, x2, y2"
        anchor_bbox_area = [128*128, 256*256, 512*512]
        anchor_bbox_ratio = [[1, 1], [2, 1], [1, 2]]
        
        img_height, img_width = image_size
        grid_height, grid_width = grid_size
        
        x_range = torch.range(0, img_width, img_width / grid_width / 2)[range(1, grid_width * 2, 2)]
        y_range = torch.range(0, img_height, img_height / grid_height / 2)[range(1, grid_height * 2, 2)]
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
        anchor_boxes[..., 0] -= anchor_boxes[..., 2] / 2
        anchor_boxes[..., 1] -= anchor_boxes[..., 3] / 2

        anchor_boxes[..., 2:] += anchor_boxes[..., :2]
        
        
        return anchor_boxes
    
    
    def pos_neg_sampling(self, anchor:torch.Tensor, target:torch.Tensor, random_choice:int = 128):
        
        anchor = anchor.clone().cpu().numpy()
        target = target.clone().cpu().numpy()
        
        _ious = ious(target, anchor)
        pos_target, pos_anc = np.where(_ious > 0.7)
        _, neg_anc = np.where(_ious < 0.3)
        negative_pedding = 0
        if len(pos_anc) > 128:
            positive_sample_index = np.random.choice(range(len(pos_anc)), random_choice, replace = False)
        else :
            positive_sample_index = np.arange(len(pos_anc))
            negative_pedding = random_choice - len(pos_anc)
            
        negative_sample_index = np.random.choice(range(len(neg_anc)), random_choice + negative_pedding, replace = False)

        return pos_target[positive_sample_index], pos_anc[positive_sample_index], neg_anc[negative_sample_index]

    
    def forward(self, x, img_size, target = None):
        batch = x.shape[0]
        grid_x = x.shape[-1]
        grid_y = x.shape[-2]
        height, width = img_size
        x = self.conv(x)
        bbox_regressor = self.bbox_layer(x) # dx, dy, dw, dh || [gird_y, grid_x, 9 * 4]
        objectness = self.score_layer(x) # [gird_y, grid_x, 9 * 2]

        bbox_regressor = bbox_regressor.view(-1, 4) # dcx, dcy, dw, dh
        objectness = objectness.view(-1, 2) # objectness
        anchor_box = self.make_anchor_box(img_size, [grid_y, grid_x]).cuda()
        anchor_box = anchor_box.view(-1, 4)
        bbox =self.box_decoder(bbox_regressor, anchor_box) # [dcx, dcy, dw, dh] + anchor = cx, cy, w, h
        bbox, objectness_bbox = self.proposal_filter(bbox, objectness, img_size)
        
        loss = None

        if self.training:
            if target is not None:
                target = target.view(-1, 4)
                # bbox_regressor = bbox_regressor.view(-1, 4)
                # score_proposal = score_proposal.view(-1, 2)
                
                # RPN 학습을 위해서 1:1 비율로 샘플링
                pos_target_idx, pos_anc_idx, neg_anc_idx = self.pos_neg_sampling(anchor_box, target)
                pos_neg_idx = np.concatenate([pos_anc_idx, neg_anc_idx])

                bbox_regressor = bbox_regressor[pos_anc_idx]
                pos_anchor_box = anchor_box[pos_anc_idx]
                objectness = objectness[pos_neg_idx] # positive, negative objectness

                target = target[pos_target_idx]
                target_objectness = torch.zeros_like(objectness)
                target_objectness[:len(pos_anc_idx), 0] = 1
                target_objectness[len(pos_anc_idx):, 1] = 1
                target_regressor = self.box_encode(pos_anchor_box, target)

                loss = self.compute_loss(bbox_regressor, objectness, target_regressor, target_objectness)
        return bbox, objectness_bbox, loss

    def box_encode(self, bbox, anchor):
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
    

    def box_decoder(self, bbox_regressor, anchor):
        w_a = (anchor[:, 2] - anchor[:, 0])
        h_a = (anchor[:, 3] - anchor[:, 1])
        cx_a = (anchor[:, 0] + anchor[:, 2])/2
        cy_a = (anchor[:, 1] + anchor[:, 3])/2

        cx = bbox_regressor[:, 0] * w_a + cx_a
        cy = bbox_regressor[:, 1] * h_a + cy_a
        w = torch.exp(bbox_regressor[:, 2]) * w_a
        h = torch.exp(bbox_regressor[:, 3]) * h_a

        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        bbox = torch.cat([x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)], dim= 1)

        return bbox
    
    def proposal_filter(self, proposals, scores, img_size):
        height, width = img_size
        proposals = proposals.clone()
        scores = scores.clone()
        if proposals.shape[0] > 200000:
            _, topk = torch.topk(scores[:, 0], 200000)
            proposals = proposals[topk]
            scores = scores[topk]

        # 이미지 밖으로 box가 나자기 않도록 만들기
        proposals[..., [1, 3]] = torch.clamp(proposals[..., [1, 3]], max=height)
        proposals[..., [1, 3]] = torch.clamp(proposals[..., [1, 3]], min=0)
        proposals[..., [0, 2]] = torch.clamp(proposals[..., [0, 2]], max=width)
        proposals[..., [0, 2]] = torch.clamp(proposals[..., [0, 2]], min=0)
        
        # 작은 스코어 제거
        keep = scores[:, 0] > 0.2
        proposals = proposals[keep]
        scores = scores[keep]

        # box 사이즈가 너무 작으면 없애기
        area = torch.abs((proposals[..., 0] - proposals[..., 2]) * (proposals[..., 1]  - proposals[..., 3]))
        keep = area > 100
        
        proposals = proposals[keep]
        scores = scores[keep]

        # NMS
        nms_index = torchvision.ops.nms(proposals, scores[:, 0], iou_threshold= 0.7)
        nms_index = nms_index[:2000]

        bbox = proposals[nms_index]
        scores = scores[nms_index]    

        return bbox, scores


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

class MyFasterRCNN(nn.Module):
    def __init__(self, n_classes):
        super(MyFasterRCNN, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(512, n_classes)
        self.bounding_box = nn.Linear(512, 4)


        
    def forward(self, x, image_size = None, target = None):
        batch = x.shape[0]
        x = x.view(batch, -1)
        x = self.fc_layer(x)
        cls = self.classifier(x)
        bbox = self.bounding_box(x)

        loss = None

        if self.training:
            if target is not None:
                loss = self.compute_loss(bbox, cls, target['box'], target['class'], image_size)

        return cls, bbox, loss
    
    def compute_loss(self, pred_box, pred_cls, target_box, target_cls, image_size):
        h_img, w_img = image_size
        
        pred_box[:, [0, 2]] /= w_img
        pred_box[:, [1, 3]] /= h_img
        
        target_box[:, [0, 2]] /= w_img
        target_box[:, [1, 3]] /= h_img
        
        # score loss        
        score_loss = F.cross_entropy(pred_cls, target_cls)        
        
        # regression loss
        regression_loss = F.smooth_l1_loss(pred_box, target_box)
        
        loss = 0.5*score_loss + regression_loss

        return loss
    
class RoIPool(nn.Module):
    def __init__(self, output_size):
        super(RoIPool, self).__init__()
        # self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        self.output_size = output_size
        
    def make_target(self, boxes, rois, targets):
        boxes = boxes.clone().detach().cpu().numpy()
        target_copy = targets['box'].clone().detach().cpu().numpy()
        _ious = ious(boxes, target_copy)
        
        target_idx = np.argmax(_ious, axis = 1)
        
        targets['box'] = targets['box'][target_idx]
        targets['class'] = targets['class'][target_idx]

        return rois, targets
        

    def forward(self, feature, boxes, image_size, targets = None):
        image_height, image_width = image_size
        feature_height, feature_width = feature.shape[-2], feature.shape[-1]
        
        scale = min(feature_height, feature_width)/min(image_height, image_width)
        
        idx = torch.zeros(boxes.shape[0]).reshape(-1, 1).cuda()
        
        boxes = torch.concat([idx, boxes], dim = -1)
        rois = torchvision.ops.roi_pool(feature, boxes, self.output_size, scale)

        # output = []
        # boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]
        # scaled_boxes = boxes.clone()
        # scaled_boxes[:, [0, 2]] *= feature_width/image_width
        # scaled_boxes[:, [1, 3]] *= feature_height/image_height
        # for box in scaled_boxes:
        #     x1, y1, x2, y2 = box.int()
        #     output.append(self.max_pool(feature[..., y1: y2, x1:x2]))

        # rois = torch.stack(output)
        
        
        if self.training:
            if targets is not None:
                rois, targets = self.make_target(boxes, rois, targets)

        return rois, targets

def train():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    vgg = vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.cuda()[:-1]
    rpn = RPN(512).cuda()
    roi_pool = RoIPool((2, 2)).cuda()
    roi_head = MyFasterRCNN(2).cuda()

    img = cv2.imread("/home/kist/datasets/VOCdevkit/VOC2012/JPEGImages/2007_002597.jpg")
        
    h, w, _ = img.shape
    scale = 600/min(h, w)

    target_bboxes, target_classes = VOC_bbox("/home/kist/datasets/VOCdevkit/VOC2012/Annotations/2007_002597.xml")
    target_labels = torch.tensor([[0, 1],
                                  [1, 0]]).float().cuda()
    target_bboxes = torch.tensor(target_bboxes).cuda()

    target_bboxes *= scale
    
    
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    h, w, _ = img.shape
    img = torch.tensor(img)
    img = img[:, :, [2, 1, 0]]/ 255.
    img = img.permute(2, 0, 1)
    
    img = transforms.Normalize(mean, std)(img).float().cuda()
    img = img.unsqueeze(0)
    
    optimizer_1  = torch.optim.Adam(nn.ModuleList([vgg, rpn]).parameters(), lr = 1e-4)
    optimizer_2  = torch.optim.Adam(nn.ModuleList([vgg, roi_pool, roi_head]).parameters(), lr = 1e-4)
    optimizer_3  = torch.optim.Adam(rpn.parameters(), lr = 1e-4)
    optimizer_4  = torch.optim.Adam(nn.ModuleList([roi_pool, roi_head]).parameters(), lr = 1e-4)
    
    
    for epoch in range(800):
        vgg.train()
        rpn.train()
        roi_pool.train()
        roi_head.train()
        train_info = {}

        for layer in vgg.parameters():
            layer.requires_grad = False
        
        for layer in roi_head.parameters():
            layer.requires_grad = False
            
        for layer in roi_pool.parameters():
            layer.requires_grad = False
            
        for layer in rpn.parameters():
            layer.requires_grad = True        
        
        targets = {
            'box': copy.deepcopy(target_bboxes),
            'class' : copy.deepcopy(target_labels)
        }
        vgg_feature = vgg(img)
        region_proposal, scores, loss = rpn(vgg_feature, (h, w), targets['box'])
        loss.backward()
        optimizer_1.step()
        optimizer_1.zero_grad()
        
        train_info['rpn_loss'] = loss
        # print(f"{epoch} : rpn_loss={loss}")

        # freezing rpn
        for layer in vgg.parameters():
            layer.requires_grad = False
        
        for layer in roi_head.parameters():
            layer.requires_grad = True
            
        for layer in roi_pool.parameters():
            layer.requires_grad = True
            
        for layer in rpn.parameters():
            layer.requires_grad = False
            

        targets = {
            'box': copy.deepcopy(target_bboxes),
            'class' : copy.deepcopy(target_labels)
        }
        vgg_feature = vgg(img)
        region_proposal, scores, loss = rpn(vgg_feature, (h, w), targets['box'])
            
        rois, targets = roi_pool(vgg_feature, region_proposal, (h, w), targets)
        b_class, bbox, loss = roi_head(rois, (h, w), targets)
        
        loss.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()
        
        train_info['roi_loss'] = loss
        # print(f"{epoch} : roi_loss={loss}")
        
        # freezing vgg
        for layer in vgg.parameters():
            layer.requires_grad = False
        
        for layer in roi_head.parameters():
            layer.requires_grad = False
            
        for layer in roi_pool.parameters():
            layer.requires_grad = False
            
        for layer in rpn.parameters():
            layer.requires_grad = True
        
        targets = {
            'box': copy.deepcopy(target_bboxes),
            'class' : copy.deepcopy(target_labels)
        }
        vgg_feature = vgg(img)
        region_proposal, _, loss = rpn(vgg_feature, (h, w), targets['box'])
        
        train_info['rpn_finetune_loss'] = loss
        # print(f"{epoch} : rpn_finetue_loss={loss}")

        loss.backward()
        optimizer_3.step()
        optimizer_3.zero_grad()
        
        # freezing vgg & rpn
        for layer in vgg.parameters():
            layer.requires_grad = False
        
        for layer in roi_head.parameters():
            layer.requires_grad = True
            
        for layer in roi_pool.parameters():
            layer.requires_grad = True
            
        for layer in rpn.parameters():
            layer.requires_grad = False
            
        roi_pool.train()
        roi_head.train()
        
        targets = {
            'box': copy.deepcopy(target_bboxes),
            'class' : copy.deepcopy(target_labels)
        }
        vgg_feature = vgg(img)
        region_proposal, _, _ = rpn(vgg_feature, (h, w))
            
        rois, targets = roi_pool(vgg_feature, region_proposal, (h, w), targets)
        b_class, bbox, loss = roi_head(rois, (h, w), targets)

        train_info['roi_finetue_loss'] = loss
        # print(f"{epoch} : roi_fintue_loss={loss}")

        loss.backward()
        optimizer_4.step()
        optimizer_4.zero_grad()
        
        print('[epcoh:{0}] rpn_loss : {1}, roi_head_loss : {2}, rpn_finetune_loss : {3}, roi_finetue_loss : {4}'.format(epoch, *train_info.values()))
        
            
    torch.save(vgg.state_dict(), "vgg.pth")
    torch.save(rpn.state_dict(), "rnp.pth")
    torch.save(roi_head.state_dict(), "roi_head.pth")
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