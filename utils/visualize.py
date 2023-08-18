import numpy as np
import cv2
from bbox import VOC_bbox

def visualize_bbox(image, bboxes, labels, scores):
    for bbox, label, score in zip(bboxes, labels, scores):
        bbox = list(map(int, bbox))
        cv2.rectangle(image, bbox[:2], bbox[2:], color= (0, 0, 255))
        cv2.putText(image, f"{label} {score}", [bbox[0], bbox[1] - 5], fontFace=0, fontScale=0.5, color=(0, 255, 0), thickness=1)
    
    return image


img = cv2.imread("d:\\datasets\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000925.jpg")
bboxes, labels = VOC_bbox("d:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_000925.xml")
scores = [100] * len(bboxes)
print(bboxes, labels)

img = visualize_bbox(img, bboxes, labels, scores)

cv2.imshow("test", img)
cv2.waitKey(10000)