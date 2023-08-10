import xml.etree.ElementTree as elemTree


def VOC_bbox(file_path):
    tree = elemTree.parse(file_path)
    root = tree.getroot()
    bboxes = []
    for child in root:
        if child.tag == 'object':
            bbox = []

            for v in child.find('bndbox'):
                bbox.append(float(v.text)) # x1, y1, x2, y2
            
            bboxes.append(bbox)

    return bboxes

if __name__ == "__main__":
    sample = "D:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_000033.xml"
    VOC_bbox(sample)