import torch
import torch.nn as nn
import torch.nn.functional as F
from roi_pooling_cuda.roi_pooling_pure_torch import RoIPooling2D


def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


class VGG16RoIHead(nn.Module):

    def __init__(self, classifier, n_class, roi_output_size):
        super().__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_output_size = roi_output_size
        self.RoI = RoIPooling2D(self.roi_output_size, 1./16)  # vgg is 1/16

    def forward(self, feature_map, rois, roi_indices):
        """Forward the chain.

         We assume that there are :math:`N` batches.

         Args:
             feature_map (Tensor): 4D feature_map tensor.
             rois (LongTensor): A bounding box array containing coordinates of
                 proposal boxes.  This is a concatenation of bounding box
                 arrays from multiple images in the batch.
                 Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                 RoIs from the :math:`i` th image,
                 :math:`R' = \\sum _{i=1} ^ N R_i`.
             roi_indices (LongTensor): An array containing indices of images to
                 which bounding boxes correspond to. Its shape is :math:`(R',)`.

         """
        rois_and_indices = torch.cat((roi_indices[:, None], rois), dim=1)
        # now rois is (R, 5)
        fixed_feature_map = self.RoI(feature_map, rois_and_indices)
        flatted = fixed_feature_map.view(fixed_feature_map.size(0), -1)
        fc7 = self.classifier(flatted)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = F.softmax(self.score(fc7), dim=1)
        return roi_cls_locs, roi_scores




