from trainer import train_model, validate, visualize
from fast_rcnn_model import FastRCNN, predict
from fast_rcnn_model_vgg import VGG16RoIHead
from fast_rcnn_model_resnet import ResNet101RoIHead
from extract_vgg import extract_vgg16
from extract_resnet import extract_resnet101
from datasets import build_dataloader, build_testloader
from region_proposal import RegionProposal
from evaluate import eval_detection_voc

import torch
import torchvision.models as models


def start_train():
    # extractor, classifier = extract_vgg16(models.vgg16(pretrained=True))
    # head = VGG16RoIHead(classifier, 3, (7, 7))

    extractor, classifier = extract_resnet101(models.resnet101(pretrained=True))
    head = ResNet101RoIHead(classifier, 3)

    model = FastRCNN(extractor, head)
    dataloader = build_dataloader(table_only=False)
    testloader = build_testloader(table_only=False)
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # next time adjust the target creator threshold
    model = train_model(model, device, dataloader, testloader)
    torch.save(model.state_dict(), "resnet101_params_0.5_0.0_target.pkl")


def start_validate():
    extractor, classifier = extract_vgg16(models.vgg16(pretrained=False))
    head = VGG16RoIHead(classifier, 2, (7, 7))

    # extractor, classifier = extract_resnet101(models.resnet101(pretrained=True))
    # head = ResNet101RoIHead(classifier, 3)

    model = FastRCNN(extractor, head)
    device = torch.device("cuda:0")
    # params_without_mo.pkl
    model = model.to(device)
    params = torch.load("params_tableonly_0.7_0.0_target.pkl")
    model.load_state_dict(params)
    testloader = build_testloader(table_only=True)
    # print(validate(model, testloader, device, iou_thresh=0.6))
    visualize(model, testloader, device)

if __name__ == '__main__':
    # start_train()
    start_validate()