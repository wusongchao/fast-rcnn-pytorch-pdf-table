import torch
import torch.nn as nn
import torch.nn.functional as F


# class RoI_pooling(torch.autograd.Function):
#     def __init__(self, output_size):
#         super().__init__()
#         self.out_w, self.out_h = output_size
#
#     def forward(ctx, feature_map, rois):
#         """
#         :param feature_map (1, C, H, W):
#         :param rois: (N, 4), 4 represent (ltx, lty, rbx, rby)
#         :return:
#         """
#         output = []
#         rois_num = rois.size(1)
#
#         for i in range(rois_num):
#             roi = rois[0][i]
#             ltx, lty, rbx, rby = roi
#             print(feature_map[:, :, lty:rby, ltx:rbx])
#             output.append(F.adaptive_max_pool2d(feature_map[:, :, lty:rby, ltx:rbx], size))
#
#         return torch.cat(output)
#
#     def backward(ctx, grad_outputs):


# class RoIPooling(nn.Module):
#     def forward(self):
#         pass


def roi_pooling(feature_map, rois, output_size, spatial_scale=1./16):
    """
    :param feature_map: (B, C, H, W)
    :param rois: (N, 5) N refers to bbox num, 5 represent (idx, ltx, lty, rbx, rby) 
    :param output_size: output size
    :param spatial_scale: feature_map scale, vgg is 1/16
    :return: (N, C, size[0], size[1])
    """
    output = []
    rois = rois.float()
    rois[:, 1:].mul_(spatial_scale)
    rois = rois.int()
    rois_num = rois.size(0)

    for i in range(rois_num):
        roi = rois[i]
        idx, ltx, lty, rbx, rby = roi
        idx = idx.item()
        output.append(F.adaptive_max_pool2d(feature_map.narrow(0, idx, 1)[..., lty:(rby+1), ltx:(rbx+1)], output_size))

    return torch.cat(output)


class RoIPooling2D(nn.Module):

    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois):
        return roi_pooling(feature_map, rois, self.output_size, self.spatial_scale)


if __name__ == '__main__':
    test_tensor = torch.tensor([
        [0.88, 0.44, 0.14, 0.16, 0.37, 0.77, 0.96, 0.27],
        [0.19, 0.45, 0.57, 0.16, 0.63, 0.29, 0.71, 0.70],
        [0.66, 0.26, 0.82, 0.64, 0.54, 0.73, 0.59, 0.26],
        [0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25],
        [0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48],
        [0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32],
        [0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48],
        [0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91]
    ])
    test_tensor = test_tensor.view(1, 1, 8, 8)

    # test_tensor.requires_grad_(True)
    rois = torch.tensor([[0, 0, 3, 6, 7], [0, 0, 3, 6, 7]])
    # rois.requires_grad_(True)
    # output = chainer.functions.roi_pooling_2d(test_tensor.numpy(), rois.numpy(), 2, 2, 1.)
    # print(output)
    output = roi_pooling(test_tensor, rois, (2, 2), spatial_scale=1)
    # output.backward(output.data.clone().uniform_())
    print(output)
    # print(test_tensor.grad)
    # print(test_tensor[test_tensor == output])
    # input = torch.randn((1,1,10,10), requires_grad=True)
    # rois = torch.tensor([[0,1,2,7,8],[0,3,3,8,8]], dtype=torch.int64, requires_grad=False)
    # #rois = ag.Variable(torch.LongTensor([[0,3,3,8,8]]),requires_grad=False)
    #
    # out = F.adaptive_max_pool2d(input, (3, 3))
    # out.backward(out.data.clone().uniform_())
    #
    # print(out)
    # print(back)