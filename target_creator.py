import torch
import numpy as np
import cv2 as cv
import time

from bbox_util import *
from region_proposal import RegionProposal
from datasets import build_dataloader, build_testloader

class ProposalTargetCreator:
    """Assign ground truth bounding boxes to given RoIs.
    
    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.
    
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.
    
    Args:
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.
    
    """

    def __init__(self, n_sample=128, pos_ratio=0.25,
                 pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh  # positive threshold
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes.

        Offsets and scales of bounding boxes are calculated using
        :func:`bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (tensor): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)` (ltx, lty, rbx, rby)
            bbox (tensor): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (tensor): Ground truth bounding box labels. Its shape
                is :math:`(R', 1)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bounding boxes.
            loc_normalize_std (tuple of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (tensor, tensor, tensor):
            
            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(R, 4)`. (since we will sample all roi
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(R, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(R, 1)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        bbox_num = bbox.size(0)
        roi = torch.cat((roi, bbox), dim=0)
        iou = bbox_iou(roi, bbox)
        # max_iou refers to the RoI's iou value with the highest iou with the gt
        max_iou, gt_assignment = iou.max(dim=1)
        # each roi's mapping gt's label
        gt_roi_label = label[gt_assignment]

        shuffle_foreground_roi_num = round(self.n_sample * self.pos_ratio)

        max_iou_numpy = max_iou.cpu().numpy()

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        foreground_roi_index = np.where(max_iou_numpy >= self.pos_iou_thresh)[0]
        foreground_roi_num = min(foreground_roi_index.size, shuffle_foreground_roi_num)

        if foreground_roi_index.size != 0:
            foreground_roi_index = np.random.choice(
                foreground_roi_index, size=foreground_roi_num, replace=False
            )

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        background_roi_index = np.where((max_iou_numpy < self.neg_iou_thresh_hi) &
                             (max_iou_numpy >= self.neg_iou_thresh_lo))[0]
        background_roi_num = min(background_roi_index.size, self.n_sample - shuffle_foreground_roi_num)

        if background_roi_index.size != 0:
            background_roi_index = np.random.choice(
                background_roi_index, size=background_roi_num, replace=False
            )

        # since each roi has already mapped to a gt, if the roi is positive, the label
        # is the same as the gt's label
        # if the roi is negative, assign 0
        keep_index = np.append(foreground_roi_index, background_roi_index)

        # gt_roi_label[:, foreground_roi_index] = gt_roi_label[:, foreground_roi_index]
        gt_roi_label[background_roi_index] = 0  # negative labels --> 0
        gt_roi_label = gt_roi_label[keep_index]

        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - torch.tensor(loc_normalize_mean, dtype=torch.float32)
                       ) / torch.tensor(loc_normalize_std, dtype=torch.float32))

        return sample_roi, gt_roi_loc.view(sample_roi.size()), gt_roi_label


if __name__ == '__main__':
    # roi = torch.tensor([[1, 1, 3, 3], [5, 5, 6, 6], [0, 1, 4, 3]])
    # gt = torch.tensor([[2, 2, 4, 4], [1, 1, 3, 3], [2, 2, 4, 4]])
    # label = torch.tensor([1, 2, 1])
    # # roi = roi.view(1, -1, 4)
    # # gt = gt.view(1, -1, 4)
    # # label = label.view(1, -1, 1)
    # proposal_target_creator = ProposalTargetCreator(pos_iou_thresh=0.3)
    # print(proposal_target_creator(roi, gt, label))
    proposal_generator = RegionProposal({}, {})
    proposal_target_creator = ProposalTargetCreator()
    dataloader = build_testloader()

    for i, data in enumerate(dataloader, 0):
        tensor, origin_image, boxes, labels = data
        rois, roi_indices = proposal_generator(origin_image)
        # since only support one image per epoch
        boxes = boxes[0]
        labels = labels[0]
        sample_rois, gt_roi_loc, gt_roi_label = proposal_target_creator(rois, boxes, labels)
        image_for_show = (origin_image.cpu().numpy())[0]
        for j in range(sample_rois.size(0)):
            if gt_roi_label[j].item() != 0:
                box = sample_rois[j]
                cv.rectangle(image_for_show, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 1)
                cv.putText(image_for_show, str(gt_roi_label[j].item()), (box[0].item(), box[1].item()), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv.imshow("", image_for_show)
        cv.waitKey(0)


