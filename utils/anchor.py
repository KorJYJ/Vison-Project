import math
import numpy as np

anchor_bbox_area = [128, 256, 512]
anchor_bbox_ratio = [[1, 1], [2, 1], [1, 2]]

def make_anchor_box(image_size : list = [224, 224], gride_size : list = [7, 7]):
    bboxes = []
    y_index, x_index = gride_size
    y_scale = image_size[0] / gride_size[0]
    x_scale = image_size[1] / gride_size[1]

    for y in range(y_index):
        for x in range(x_index):
            for area in anchor_bbox_area:
                for ratio in anchor_bbox_ratio:
                    v = math.sqrt(area / (ratio[0] * ratio[1]))
                    w = ratio[0] * v
                    h = ratio[1] * v

                    cx = (2*x+1) / 2 * x_scale
                    cy = (2*y+1) / 2 * y_scale
                    bbox = [cx, cy, w, h]
                    bboxes.append(bbox)
        
    bboxes = np.array(bboxes).reshape(gride_size[0], gride_size[1], 9, 4)

    return bboxes


# anchor_boxes = [
#     [8, 16], [16, 8], [11.3, 11.3],
#     [11.3, 22.6], [22.6, 11.3], [16, 16],
#     [16, 32],[32, 16], [22.6, 22.6]
#     ]

# def make_anchor_box(image_size : list = [224, 224], gride_size : list = [7, 7]):
#     bboxes = []
#     y_index, x_index = gride_size
#     y_scale = image_size[0] / gride_size[0]
#     x_scale = image_size[1] / gride_size[1]

#     for y in range(y_index):
#         for x in range(x_index):
#             for anch in anchor_boxes:
#                 w, h = anch
#                 cx = (2*x+1) / 2 * x_scale
#                 cy = (2*y+1) / 2 * y_scale
#                 bbox = [cx, cy, w, h]
#                 bboxes.append(bbox)
    
#     bboxes = np.array(bboxes).reshape(gride_size[0], gride_size[1], 9, 4)

#     return bboxes