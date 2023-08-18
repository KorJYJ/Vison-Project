import xml.etree.ElementTree as elemTree


def VOC_bbox(file_path):
    tree = elemTree.parse(file_path)
    root = tree.getroot()
    bboxes = []
    names = []
    for child in root:
        if child.tag == 'object':
            bbox = []
            
            name = child.find('name').text

            for v in child.find('bndbox'):
                bbox.append(float(v.text)) # x1, y1, x2, y2
            
            bboxes.append(bbox)
            names.append(name)
            
    return bboxes, names

if __name__ == "__main__":
    sample = "D:\\datasets\\VOCdevkit\\VOC2012\\Annotations\\2007_000032.xml"
    bboxes, names = VOC_bbox(sample)
    print(bboxes)
    print(names)