import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv

from region_proposal import RegionProposal, LineBaseRegionProposal
from target_creator import ProposalTargetCreator
from fast_rcnn_model import predict
from utils_func import get_tensor_bytes
from evaluate import eval_detection_voc
from visdom import Visdom
import numpy as np
viz=Visdom()

def get_optimizer(model, base_lr, momentum=0.9,
                  weight_regularization=0.0005, bias_regularization=0):
    lr = base_lr
    parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                parameters.append({
                    "params": [value],
                    "lr": lr * 2,
                    "weight_decay": bias_regularization
                })
            else:
                parameters.append({
                    "params": [value],
                    "lr": lr,
                    "weight_decay": weight_regularization
                })

    optimizer = optim.SGD(parameters, momentum=momentum)
    return optimizer


def loc_loss(criterion, n_sample, roi_cls_loc, gt_roi_loc, gt_roi_label, device=None):
    roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)

    # for each sample, select the gt class
    roi_loc = roi_cls_loc[torch.arange(0, n_sample, dtype=torch.long, device=device), gt_roi_label.long().cuda()]
    in_weight = torch.zeros_like(gt_roi_loc).cuda()
    in_weight[(gt_roi_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    # now only the roi assign with the gt_label is foreground are selected,
    # in_weight is a mask array
    return criterion(roi_loc * in_weight, gt_roi_loc * in_weight)

# 24 17
# 8 5
def train_model(model, device, dataloader, testloader, num_epochs=24):
    # proposal_generator = RegionProposal({}, {})
    proposal_generator = LineBaseRegionProposal({}, {})
    proposal_target_creator = ProposalTargetCreator(pos_iou_thresh=0.5, neg_iou_thresh_lo=0.0)

    model = model.to(device)
    optimizer = get_optimizer(model, 0.001, weight_regularization=0.0005)
    # print(len(dataloader))

    # pay attention to the doc,
    # Sets the learning rate of each parameter group to the initial lr "times" a given function
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 if epoch < 17 else 0.1)

    class_criterion = nn.CrossEntropyLoss()
    bbox_regression_criterion = nn.SmoothL1Loss()

    model.train()
    win_line = viz.line(X=np.arange(10),Y=np.arange(10))
    maps = []

    losses = []
    table_aps=[]
    figure_aps=[]
    epochs=[]

    for epoch in range(num_epochs):
        scheduler.step()
        # lr = []
        # for param_group in optimizer.param_groups:
        #     lr += [param_group['lr']]
        # print(lr)

        running_loss = 0.

        for i, data in enumerate(dataloader, 0):
            tensor, origin_image, boxes, labels = data
            optimizer.zero_grad()

            rois, roi_indices = proposal_generator(origin_image)
            # since only support one image per epoch
            boxes = boxes[0]
            labels = labels[0]
            sample_rois, gt_roi_loc, gt_roi_label = proposal_target_creator(rois, boxes, labels)
            n_sample = sample_rois.size(0)
            roi_indices = torch.zeros(n_sample, dtype=torch.int)

            tensor = tensor.to(device)
            sample_rois = sample_rois.to(device)
            roi_indices = roi_indices.to(device)
            gt_roi_loc = gt_roi_loc.to(device)
            gt_roi_label = gt_roi_label.long()
            gt_roi_label = gt_roi_label.to(device)
            roi_cls_loc, roi_scores = model(tensor, sample_rois, roi_indices)

            class_loss = class_criterion(roi_scores, gt_roi_label)
            regression_loss = loc_loss(bbox_regression_criterion, n_sample,
                                       roi_cls_loc, gt_roi_loc, gt_roi_label, device)

            loss = class_loss + regression_loss
            loss.backward()
            optimizer.step()

            print(i)

            running_loss += loss.item()

            if i % 300 == 299:
                validate_result = training_validate(model, testloader, device, 0.6)
                print(validate_result)
                # loss table_ap = dict[ap][1] figure_ap = dict[ap][2] map = dict[map]
                maps.append(validate_result["map"])
                losses.append(running_loss/300.)
                table_aps.append(validate_result["ap"][1])
                figure_aps.append(validate_result["ap"][2])
                epochs.append(epoch*len(dataloader)+i)
                viz.line(
                    X=np.column_stack((epochs, epochs, epochs, epochs)),
                    Y=np.column_stack((maps, losses, table_aps, figure_aps)),
                    win=win_line,
                    opts=dict(
                        legend=["maps", "losses", "table_ap", "figure_ap"]
                    )
                )
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 300))
                running_loss = 0.

    return model


def training_validate(model, dataloader, device, iou_thresh):
    proposal_generator = LineBaseRegionProposal({}, {})

    # model = model.to(device)

    model.eval()

    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    gt_bboxes = []
    gt_labels = []

    # dataiter = iter(dataloader)

    with torch.no_grad():
        for tensor, origin_image, gt_box, gt_label in dataloader:
            tensor = tensor.to(device)
            gt_box = gt_box[0]
            gt_label = gt_label[0]

            bbox, label, score = predict(model, proposal_generator, tensor, origin_image)

            pred_bboxes.append(bbox.cpu().numpy())
            pred_labels.append(label.cpu().numpy())
            pred_scores.append(score.cpu().numpy())
            gt_bboxes.append(gt_box.numpy())
            gt_labels.append(gt_label.numpy())

    return eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=iou_thresh, use_07_metric=True)


def validate(model, dataloader, device, iou_thresh):
    proposal_generator = LineBaseRegionProposal({}, {})

    model = model.to(device)

    model.eval()

    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    gt_bboxes = []
    gt_labels = []

    with torch.no_grad():
        for tensor, origin_image, gt_box, gt_label in dataloader:
            tensor = tensor.to(device)
            gt_box = gt_box[0]
            gt_label = gt_label[0]

            bbox, label, score = predict(model, proposal_generator, tensor, origin_image)

            pred_bboxes.append(bbox.cpu().numpy())
            pred_labels.append(label.cpu().numpy())
            pred_scores.append(score.cpu().numpy())
            gt_bboxes.append(gt_box.numpy())
            gt_labels.append(gt_label.numpy())

    return eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=iou_thresh, use_07_metric=True)


def visualize(model, dataloader, device):
    proposal_generator = LineBaseRegionProposal({}, {})

    model = model.to(device)

    model.eval()

    with torch.no_grad():
        for tensor, origin_image, gt_box, gt_label in dataloader:
            tensor = tensor.to(device)
            bbox, label, score = predict(model, proposal_generator, tensor, origin_image)

            image_for_show = (origin_image.cpu().numpy())[0]
            for i in range(bbox.size(0)):
                box = bbox[i]
                if label[i] != 0:
                    cv.rectangle(image_for_show, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()),
                                 (0, 0, 255), 1)
                    cv.putText(image_for_show, str(label[i].item()), (box[0].item(), box[1].item()),
                               cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            cv.imshow("", image_for_show)
            cv.waitKey(0)

    return


if __name__ == '__main__':
    # lr_func = lambda epoch: 0.001 if epoch < 50 else 0.0005
    # model = nn.Sequential(
    #     nn.Linear(5, 2),
    #     nn.ReLU()
    # )
    # optimizer = get_optimizer(model, 0.001)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    # print(optimizer.param_groups)
    # for epoch in range(100):
    #     scheduler.step()
    #     print(optimizer.param_groups)

    # features = torch.randn(2, 7)
    # gt = torch.tensor([1, 1])
    # model = nn.Sequential(
    #     nn.Linear(7, 4),
    #     nn.ReLU(),
    #     nn.Linear(4, 4)
    # )
    # optimizer = optim.SGD(model.parameters(), lr=0.005)
    # f = nn.CrossEntropyLoss()
    #
    # print(features)
    # print(model(features))
    #
    # for epoch in range(1000):
    #     optimizer.zero_grad()
    #     output = model(features)
    #     loss = f(output, gt)
    #     loss.backward()
    #     optimizer.step()
    #
    # print(model(features))
    pass