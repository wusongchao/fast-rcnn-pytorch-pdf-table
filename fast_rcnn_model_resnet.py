import torch
import torch.nn as nn
import torch.nn.functional as F
from roi_pooling_cuda.roi_pooling_pure_torch import RoIPooling2D


def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


class ResNet101RoIHead(nn.Module):

    def __init__(self, classifier, n_class, roi_output_size=(14, 14)):
        super().__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_output_size = roi_output_size
        self.RoI = RoIPooling2D(self.roi_output_size, 1./16)  # vgg is 1/16

    def forward(self, feature_map, rois, roi_indices):
        rois_and_indices = torch.cat((roi_indices[:, None], rois), dim=1)
        # now rois is (R, 5)
        fixed_feature_map = self.RoI(feature_map, rois_and_indices)
        fc7 = self.classifier(fixed_feature_map)
        flatted = fc7.view(fc7.size(0), -1)
        roi_cls_locs = self.cls_loc(flatted)
        roi_scores = F.softmax(self.score(flatted), dim=1)
        return roi_cls_locs, roi_scores
