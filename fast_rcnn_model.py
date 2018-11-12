import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import datasets
import region_proposal
from nms import non_maximum_suppression
from bbox_util import loc2bbox


class FastRCNN(nn.Module):
    """Base class for Fast R-CNN.

    This is a base class for Fast R-CNN links supporting object detection
    API [#]_. The following three stages constitute Fast R-CNN.

    1. **Region Proposal**: Given the origin image, \ 
        produce set of RoIs around objects.
    2. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    3. **RoI pooling**: each RoI is mapping to the feature map, each RoI \
        will be converted to a fix size feature map by the pooling layer
    4. **Localization and Classification Heads**: Using feature map vector that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    Tsg and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """
    def __init__(self, extractor, head,
                loc_normalize_mean=(0., 0., 0., 0.),
                loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
    ):
        super().__init__()
        self.extractor = extractor
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        # self.use_preset('evaluate')

    def n_class(self):
        return self.head.n_class

    def forward(self, image_tensor, rois, roi_indices):
        """Forward Fast R-CNN.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            image_tensor (Tensor): 4D image tensor(BCHW).
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            

        Returns:
            Tensor, Tensor, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) * 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        # img_size = image_tensor.shape[2:]
        feature_maps = self.extractor(image_tensor)
        roi_cls_locs, roi_scores = self.head(feature_maps, rois, roi_indices)
        return roi_cls_locs, roi_scores

    # def use_preset(self, preset):
    #     """Use the given preset during prediction.
    #
    #     This method changes values of :obj:`self.nms_thresh` and
    #     :obj:`self.score_thresh`. These values are a threshold value
    #     used for non maximum suppression and a threshold value
    #     to discard low confidence proposals in :meth:`predict`,
    #     respectively.
    #
    #     If the attributes need to be changed to something
    #     other than the values provided in the presets, please modify
    #     them by directly accessing the public attributes.
    #
    #     Args:
    #         preset ({'visualize', 'evaluate'): A string to determine the
    #             preset to use.
    #
    #     """
    #     if preset == 'visualize':
    #         self.nms_thresh = 0.3
    #         self.score_thresh = 0.7
    #     elif preset == 'evaluate':
    #         self.nms_thresh = 0.3
    #         self.score_thresh = 0.05
    #     else:
    #         raise ValueError('preset must be visualize or evaluate')

    # def _suppress(self, raw_cls_bbox, raw_prob):
    #     bbox = []
    #     label = []
    #     score = []
    #     # skip cls_id = 0 because it is the background class
    #     for l in range(1, self.n_class()):
    #         cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
    #         prob_l = raw_prob[:, l]
    #         mask = prob_l > self.score_thresh
    #         cls_bbox_l = cls_bbox_l[mask]
    #         prob_l = prob_l[mask]
    #         keep = non_maximum_suppression(
    #             cp.array(cls_bbox_l), self.nms_thresh, prob_l)
    #         keep = cp.asnumpy(keep)
    #         bbox.append(cls_bbox_l[keep])
    #         # The labels are in [0, self.n_class - 2].
    #         label.append((l - 1) * np.ones((len(keep),)))
    #         score.append(prob_l[keep])
    #     bbox = np.concatenate(bbox, axis=0).astype(np.float32)
    #     label = np.concatenate(label, axis=0).astype(np.int32)
    #     score = np.concatenate(score, axis=0).astype(np.float32)
    #     return bbox, label, score
    #
    # def predict(self, image_tensor, visualize=False):
    #     """Detect objects from images.
    #
    #     This method predicts objects for each image.
    #
    #     Args:
    #         image_tensor (iterable of Tensor): Arrays holding images.
    #             preprocessed(imageNet Normalize) (N, C, H, W) Tensor
    #
    #     Returns:
    #        tuple of lists:
    #        This method returns a tuple of three lists,
    #        :obj:`(bboxes, labels, scores)`.
    #
    #        * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
    #            where :math:`R` is the number of bounding boxes in a image. \
    #            Each bouding box is organized by \
    #            :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
    #            in the second axis.
    #        * **labels** : A list of integer arrays of shape :math:`(R,)`. \
    #            Each value indicates the class of the bounding box. \
    #            Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
    #            number of the foreground classes.
    #        * **scores** : A list of float arrays of shape :math:`(R,)`. \
    #            Each value indicates how confident the prediction is.
    #
    #     """
    #     self.eval()
    #     if visualize:
    #         self.use_preset('visualize')
    #
    #     bboxes = []
    #     labels = []
    #     scores = []
    #
    #     image_num = img
    #     for img in prepared_imgs:
    #         img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
    #         scale = img.shape[3] / size[1]
    #         roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
    #         # We are assuming that batch size is 1.
    #         roi_score = roi_scores.data
    #         roi_cls_loc = roi_cls_loc.data
    #         roi = at.totensor(rois) / scale
    #
    #         # Convert predictions to bounding boxes in image coordinates.
    #         # Bounding boxes are scaled to the scale of the input images.
    #         mean = t.Tensor(self.loc_normalize_mean).cuda(). \
    #             repeat(self.n_class)[None]
    #         std = t.Tensor(self.loc_normalize_std).cuda(). \
    #             repeat(self.n_class)[None]
    #
    #         roi_cls_loc = (roi_cls_loc * std + mean)
    #         roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
    #         roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
    #         cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
    #                             at.tonumpy(roi_cls_loc).reshape((-1, 4)))
    #         cls_bbox = at.totensor(cls_bbox)
    #         cls_bbox = cls_bbox.view(-1, self.n_class * 4)
    #         # clip bounding box
    #         cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
    #         cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
    #
    #         prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1))
    #
    #         raw_cls_bbox = at.tonumpy(cls_bbox)
    #         raw_prob = at.tonumpy(prob)
    #
    #         bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
    #         bboxes.append(bbox)
    #         labels.append(label)
    #         scores.append(score)
    #
    #     self.use_preset('evaluate')
    #     self.train()
    #     return bboxes, labels, scores


def _suppress(raw_cls_bbox, raw_prob, n_class, nms_thresh, score_thresh):
    # cpu base method
    # since the parameter "n_class" is depend on the model's "n_class"
    # model's "n_class" is defined by the head part
    # it include the background class
    bbox = []
    label = []
    score = []
    # skip cls_id = 0 because it is the background class
    for l in range(1, n_class):
        # (N, 4)
        if raw_cls_bbox.size(0) < n_class:
            # raw_cls_bbox has less num than n_class, e.g., only 1 bbox but 2 class
            cls_bbox_l = raw_cls_bbox[...]
        else:
            cls_bbox_l = raw_cls_bbox.reshape((-1, n_class, 4))[:, l, :]

        # cls_bbox_l = raw_cls_bbox[...]

        prob_l = raw_prob[:, l]
        mask = prob_l > score_thresh
        cls_bbox_l = cls_bbox_l[mask]
        prob_l = prob_l[mask]
        # NMS return the indices of keep bbox
        keep = non_maximum_suppression(cls_bbox_l, prob_l, nms_thresh)

        bbox.append(cls_bbox_l[keep])
        # The labels are in [0, self.n_class - 2].
        label.append(l * torch.ones((keep.nelement(),)))
        score.append(prob_l[keep])

    # cls_bbox_l = raw_cls_bbox[...]
    #
    # print(cls_bbox_l.size(), raw_prob.size())
    #
    # prob_l, _ = torch.max(raw_prob, dim=1)
    # mask = prob_l > score_thresh
    # cls_bbox_l = cls_bbox_l[mask]
    # prob_l = prob_l[mask]
    # keep = non_maximum_suppression(cls_bbox_l, prob_l, nms_thresh)
    #
    # bbox.append(cls_bbox_l[keep])
    # # The labels are in [0, self.n_class - 2].
    # label.append(torch.argmax(prob_l[keep], dim=1))
    # score.append(prob_l[keep])

    bbox = torch.cat(bbox, dim=0).int()
    label = torch.cat(label, dim=0).int()
    score = torch.cat(score, dim=0).float()

    return bbox, label, score


def predict(model, proposal_creator, image_tensor, image_matrix, nms_thresh=0.2, score_thresh=0.5):

    rois, roi_indices = proposal_creator(image_matrix)
    roi_cls_loc, roi_scores = model(image_tensor, rois, roi_indices)
    n_class = model.n_class()

    mean = torch.tensor(model.loc_normalize_mean).repeat(n_class)[None]
    std = torch.tensor(model.loc_normalize_std).repeat(n_class)[None]

    roi_cls_loc = roi_cls_loc.cpu()

    roi_cls_loc = (roi_cls_loc * std + mean)
    roi_cls_loc = roi_cls_loc.view(-1, n_class, 4)
    rois = rois.view(-1, 1, 4).expand_as(roi_cls_loc)
    cls_bbox = loc2bbox(rois.reshape(-1, 4), roi_cls_loc.reshape(-1, 4))  # do the bbox regression to "adjust" the bbox
    cls_bbox = cls_bbox.view(-1, 4)
    # clip bounding box
    size = image_matrix[0].shape
    cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
    cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

    prob = roi_scores
    bbox, label, score = _suppress(cls_bbox.cpu(), prob.cpu(),
                                   n_class, nms_thresh, score_thresh)

    return bbox, label, score


if __name__ == '__main__':
    # origin_model = models.resnet18(pretrained=True)
    rois = torch.randn(8, 4)
    by_contour = torch.tensor([1, 1, 0, 0, 0, 0, 0, 1])
    loc = torch.randn(8, 3, 4)
    print(rois)
    need_to_regress = rois[by_contour > 0]
    regress_loc = loc[by_contour > 0]
    need_to_regress = need_to_regress.view(-1, 1, 4).expand_as(regress_loc)
    res = loc2bbox(need_to_regress.reshape(-1, 4), regress_loc.reshape(-1, 4))
    print(res.view(-1, 4))