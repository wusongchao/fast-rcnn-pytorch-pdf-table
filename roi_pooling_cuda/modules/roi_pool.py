from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction


class RoIPooling(Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()

        self.pooled_width, self.pool_height = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
