import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

from cython_bbox import bbox_overlaps as bbox_ious
from utils.anchor import make_anchor_box
from utils.bbox import VOC_bbox

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def iou_distance(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

if __name__ == "__main__":
    sample = "D:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_000033.xml"
    bboxes = VOC_bbox(sample)
    anchor_boxes = make_anchor_box()

    bboxes = np.array(bboxes)

    x, y, n_anch, n_bbox = anchor_boxes.shape
    anchor_boxes = anchor_boxes.reshape(-1, 4)

    anchor_boxes[:, 2:] += anchor_boxes[:, :2]

    _ious = ious(bboxes, anchor_boxes)
    print(_ious)

    print(iou_distance(anchor_boxes, bboxes))

    print(np.where(_ious > 0.001))
    # print(bboxes)