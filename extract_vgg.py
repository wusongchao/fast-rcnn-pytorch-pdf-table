import torch
import torch.nn as nn
import torchvision.models as models


def extract_vgg16(model):
    """
    First, the last max pooling layer is replaced by a RoI
    pooling layer that is configured by setting H and W to be
    compatible with the net’s first fully connected layer (e.g.,
    H = W = 7 for VGG16).
    
    Second, the network’s last fully connected layer and softmax
    (which were trained for 1000-way ImageNet classification)
    are replaced with the two sibling layers described
    earlier (a fully connected layer and softmax over K + 1 categories
    and category-specific bounding-box regressors).
    
    So this function returns the feature extractor from vgg but 
    removing the last max pooling layer.
    And the classifier removing the last fc layer
    :param model: 
    :return: 
    """
    features = list(model.features.children())
    features.pop()  # remove the last max pooling layer

    classifier = list(model.classifier.children())
    classifier.pop()  # remove the last fully connected layer and softmax

    return nn.Sequential(*features), nn.Sequential(*classifier)


if __name__ == '__main__':
    features, classifier = extract_vgg16(models.vgg16(pretrained=True))
    # features, classifier = extract_resnet101(models.resnet101())
    print(features, classifier)
