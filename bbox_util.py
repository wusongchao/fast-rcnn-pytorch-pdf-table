import torch
import numpy as np


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (tensor): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`torch.float32`.
        bbox_b (tensor): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`torch.float32`.

    Returns:
        tensor:
        An tensor whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """

    bbox_a_lt = bbox_a[:, :2]
    bbox_a_rb = bbox_a[:, 2:]
    bbox_b_lt = bbox_b[:, :2]
    bbox_b_rb = bbox_b[:, 2:]

    area_a = torch.prod(bbox_a_rb - bbox_a_lt + 1, dim=1)
    area_b = torch.prod(bbox_b_rb - bbox_b_lt + 1, dim=1)

    intersect_lt = torch.max(bbox_a[:, None, :2], bbox_b_lt)
    intersect_rb = torch.min(bbox_a[:, None, 2:], bbox_b_rb)
    intersect_w_h_tensor = intersect_rb - intersect_lt + 1
    intersect_w_h_tensor = torch.max(
        intersect_w_h_tensor,
        torch.zeros_like(intersect_w_h_tensor)
    )
    intersect_area = torch.prod(intersect_w_h_tensor, dim=2)

    return intersect_area.float() / (area_a[:, None] + area_b - intersect_area).float()


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\(g_y - p_y) / p_h`
    * :math:`t_x = \\(g_x - p_x) / p_w`
    * :math:`t_h = \\log(\\g_h / p_h)`
    * :math:`t_w = \\log(\\g_w / p_w)`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (tensor): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{xmin}, p_{ymin}, p_{xmax}, p_{ymax}`.
        dst_bbox (tensor): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{xmin}, g_{ymin}, g_{xmax}, g_{ymax}`.

    Returns:
        tensor:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    """
    src_bbox = src_bbox.cpu().numpy()
    dst_bbox = dst_bbox.cpu().numpy()

    width = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    height = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0] + 1.0
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1] + 1.0
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    # to ensure that w/h is not zero(at least the minimum positive float)
    eps = np.finfo(np.float32).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = torch.tensor(np.vstack((dx, dy, dw, dh)).transpose(), dtype=torch.float32)
    return loc

def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (Tensor): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{xmin}, p_{ymin}, p_{xmax}, p_{ymax}`.
        loc (Tensor): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_x, t_y, t_w, t_h`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{xmin}, \\hat{g}_{ymin},
        \\hat{g}_{xmax}, \\hat{g}_{ymax}`.

    """

    if src_bbox.size(0) == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.float()

    src_width = src_bbox[:, 2] - src_bbox[:, 0] + 1.0
    src_height = src_bbox[:, 3] - src_bbox[:, 1] + 1.0
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_y = dy * src_height[:, None] + src_ctr_y[:, None]
    ctr_x = dx * src_width[:, None] + src_ctr_x[:, None]
    h = torch.exp(dh) * src_height[:, None]
    w = torch.exp(dw) * src_width[:, None]

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox.float()


if __name__ == '__main__':
    roi = torch.tensor([[1, 1, 3, 3], [5, 5, 6, 6], [0, 1, 4, 3]])
    gt = torch.tensor([[2, 2, 4, 4], [1, 1, 3, 3], [2, 2, 4, 4]])
    print(bbox_iou(roi, gt))