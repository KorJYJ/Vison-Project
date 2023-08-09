import cv2
import numpy as np
import torch
from layers.blocks import ResBottleBlock, ConvBlock,classifier
from sklearn import metrics
from model import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    type=str,
                    required=True)

args = parser.parse_args()

def dataloader():
    cat = cv2.imread("data/cat.jpg") 
    dog = cv2.imread("data/dog.jpg")

    cat = cv2.resize(cat, (224, 224))
    dog = cv2.resize(dog, (224, 224))

    return cat, dog

get_model = {
    "ResNet" : MyResnet
}

if __name__ == "__main__":
    ori_cat, ori_dog = dataloader()
    label = torch.tensor([[0], [1]]).cuda().float()

    cat = torch.from_numpy(ori_cat).permute(2, 0, 1).float()
    dog = torch.from_numpy(ori_dog).permute(2, 0, 1).float()

    in_img = torch.stack([cat, dog])
    model = get_model[args.model]

    epochs = 10
    criterian = torch.nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(in_img.cuda())
        loss = criterian(output, label)
        loss.backward()
        optimizer.step()

        predict = output.detach().cpu().numpy() > 0.5
        predict = predict.astype(np.int16)
        acc = metrics.accuracy_score(label.cpu().numpy(), predict)
        print("epcohs : {0}, loss : {1}, acc : {2}".format(epoch, loss, acc))
    