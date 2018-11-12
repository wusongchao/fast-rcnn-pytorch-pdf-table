import torch
import numpy as np


def non_maximum_suppression(bbox, scores, threshold):
    """Suppress bounding boxes according to their IoUs.

    This method checks each bounding box sequentially and selects the bounding
    box if the Intersection over Unions (IoUs) between the bounding box and the
    previously selected bounding boxes is less than :obj:`thresh`. This method
    is mainly used as postprocessing of object detection.
    The bounding boxes are selected from ones with higher scores.
    If :obj:`score` is not provided as an argument, the bounding box
    is ordered by its index in ascending order.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    :obj:`score` is a float array of shape :math:`(R,)`. Each score indicates
    confidence of prediction.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    an input. Please note that both :obj:`bbox` and :obj:`score` need to be
    the same type.
    The type of the output is the same as the input.

    Args:
        bbox (numpy): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        scores (numpy): An array of confidences whose shape is :math:`(R,)`.
        threshold (float): Threshold of IoUs.

    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`torch.int32`. Note that :math:`K \\leq R`.

    """
    if bbox.nelement() == 0:
        return torch.tensor([], dtype=torch.long)

    bbox_numpy = bbox.cpu().numpy()
    scores_numpy = scores.cpu().numpy()
    x1 = bbox_numpy[:, 0]
    y1 = bbox_numpy[:, 1]
    x2 = bbox_numpy[:, 2]
    y2 = bbox_numpy[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort return the index
    order = scores_numpy.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # if IoU <= threshold
        # inds size is order length - 1
        inds = np.where(ovr <= threshold)[0]
        # since inds size is order length - 1
        # remove the order's head
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long)


if __name__ == '__main__':
    dets = torch.tensor([
        [204, 102, 358, 250],
        [257, 118, 380, 250],
        [280, 135, 400, 250],
        [255, 118, 360, 235]])
    scores = torch.tensor([0.5, 0.7, 0.6, 0.7])
    threshold = 0.3
    print(non_maximum_suppression(dets, scores, threshold))