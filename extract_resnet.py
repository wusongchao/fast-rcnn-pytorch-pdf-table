import torch
import torch.nn as nn
import torchvision.models as models


def extract_resnet101(model):
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
    #                        bias=False)
    # self.bn1 = nn.BatchNorm2d(64)
    # self.relu = nn.ReLU(inplace=True)
    # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # self.layer1 = self._make_layer(block, 64, layers[0])
    # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # self.avgpool = nn.AvgPool2d(7, stride=1)
    # self.fc = nn.Linear(512 * block.expansion, num_classes)

    # x = self.conv1(x)
    # x = self.bn1(x)
    # x = self.relu(x)
    # x = self.maxpool(x)
    #
    # x = self.layer1(x)
    # x = self.layer2(x)
    # x = self.layer3(x)
    # x = self.layer4(x)
    #
    # x = self.avgpool(x)
    # x = x.view(x.size(0), -1)
    # x = self.fc(x)

    conv1 = model.conv1
    # for param in conv1.parameters():
    #     param.require_grad = False

    bn1 = model.bn1
    for param in bn1.parameters():
        param.require_grad = False  # freeze the bn layer

    relu = model.relu
    maxpool = model.maxpool
    conv2 = model.layer1
    # for param in conv2.parameters():
    #     param.require_grad = False

    conv3 = model.layer2
    # for param in conv3.parameters():
    #     param.require_grad = False

    conv4 = model.layer3

    conv5 = model.layer4
    avgpool = model.avgpool

    return nn.Sequential(conv1, bn1, relu, maxpool, conv2, conv3, conv4), \
           nn.Sequential(conv5, avgpool)