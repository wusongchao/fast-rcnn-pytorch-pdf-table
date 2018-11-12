import cv2 as cv
import os
import numpy as np
import torch
from path_constant import *
from datasets import build_testloader
import time
import random


class RegionProposal:

    def __init__(self, contour_setting, sliding_window_setting, validate=False):
        self.contour_setting = contour_setting
        self.sliding_window_setting = sliding_window_setting
        self.validate = validate

    def __call__(self, rgb_matrix):
        '''
        :param rgb_matrix: (N, H, W, C)
        :return: LongTensor rois (N, 4), LongTensor roi_indices (N, )
        '''
        rgb_matrix = rgb_matrix.numpy()
        image_num = rgb_matrix.shape[0]
        rois = []
        roi_indices = []
        for i in range(image_num):
            single_matrix = rgb_matrix[i]
            res = contour_proposal(single_matrix, **self.contour_setting)
            if self.validate:
                res += sliding_window(single_matrix, **self.sliding_window_setting)
            else:
                if len(res) == 0:
                    res = sliding_window(single_matrix, **self.sliding_window_setting)
            rois.append(torch.tensor(res, dtype=torch.int))
            roi_indices.append(torch.full((len(res),), i, dtype=torch.int))
        return torch.cat(rois), torch.cat(roi_indices)


def overlap_ratio(l1, l2, r1, r2):
    if l1 > l2:
        l1, l2 = l2, l1
    if r1 > r2:
        r1, r2 = r2, r1

    len1 = l2 - l1
    len2 = r2 - r1
    intersect1 = max(l1, r1)
    intersect2 = min(l2, r2)
    intersect_len = max(intersect2 - intersect1, 0)
    return intersect_len / (len1 + len2 - intersect_len)


def np_base_overlap_ratio(l1, l2, r1, r2):
    # must ensure that l1, l2, r1, r2 have the right order
    len1 = l2 - l1
    len2 = r2 - r1
    intersect1 = np.maximum(l1, r1)
    intersect2 = np.minimum(l2, r2)
    intersect_len = (intersect2 - intersect1).clip(0)
    return intersect_len / (len1 + len2 - intersect_len)


def too_similar(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):
    if approximately_vertical(lx1, ly1, lx2, ly2) and approximately_vertical(rx1, ry1, rx2, ry2):
        ratio = overlap_ratio(ly1, ly2, ry1, ry2)
        if ratio > 0.85 and abs(lx1 - rx1) < 6.:
            return True

    if approximately_horizontal(lx1, ly1, lx2, ly2) and approximately_horizontal(rx1, ry1, rx2, ry2):
        ratio = overlap_ratio(lx1, lx2, rx1, rx2)
        if ratio > 0.85 and abs(ly1 - ry1) < 10.:
            return True

    return False


def np_base_approximately_vertical(x1s, y1s, x2s, y2s):
    """
    
    :param x1s: (N, ) 
    :param y1s: (N, )
    :param x2s: (N, )
    :param y2s: (N, )
    :return: (N, )
    """
    return (np.abs(x1s - x2s) <= 2.) | (np.abs((y1s - y2s) / (x1s - x2s)) > 5.671)
    # if abs(x1 - x2) <= 2.:
    #     return True
    # k = abs((y1 - y2) / (x1 - x2))
    # if k > 5.671:
    #     return True
    # return False


def np_base_approximately_horizontal(x1s, y1s, x2s, y2s):
    return (np.abs(x1s - x2s) > 2.) & (np.abs((y1s - y2s) / (x1s - x2s)) < 0.176)
    # if abs(x1 - x2) <= 2.:
    #     return False
    # k = abs((y1 - y2) / (x1 - x2))
    # if k < 0.176:
    #     return True
    # return False


def np_base_too_similar(line, other_lines):
    """
    
    :param line: (4, )
    :param other_lines: (N, 4) 
    :return: bool value
    """
    lx1, ly1, lx2, ly2 = line
    rx1 = other_lines[:, 0]
    ry1 = other_lines[:, 1]
    rx2 = other_lines[:, 2]
    ry2 = other_lines[:, 3]
    b1 = approximately_vertical(lx1, ly1, lx2, ly2) and np_base_approximately_vertical(rx1, ry1, rx2, ry2)


def filter_too_similar_line(lines):
    res = []
    lines_len = len(lines)
    for i in range(lines_len):
        line = lines[i]
        lx1, ly1, lx2, ly2 = line[0]
        similar = False
        for j in range(i + 1, lines_len):
            other = lines[j]
            rx1, ry1, rx2, ry2 = other[0]
            if too_similar(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):
                similar = True
                break
        if not similar:
            res.append(line)

    return res


def np_base_filter_too_similar_line(lines):
    res = []
    lines = np.array(lines, dtype=np.float).squeeze()
    lines_num = lines.shape[0]
    for i in range(lines_num):
        for_check = lines[i]
        indices = np.arange(i + 1, lines_num)
        left = lines[indices]

    return res


def shuffle_and_squeeze_hough_line(lines):
    res = []
    for line in lines:
        line = line[0]
        x1, y1, x2, y2 = line
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        if approximately_horizontal_vertical(x1, y1, x2, y2):
            if l1 > l2:
                l1, l2 = l2, l1
            if r1 > r2:
                r1, r2 = r2, r1
    return res



def houghline_base_proposal(rgb_matrix, rho=1, theta=np.pi/180, threshold=30, min_line_length=50, max_line_gap=2):
    gray = cv.cvtColor(rgb_matrix, cv.COLOR_RGB2GRAY)
    # edges = cv.Canny(gray, 50, 150, apertureSize=3)
    edges = auto_canny(gray)
    lines = cv.HoughLinesP(edges, rho, theta, threshold,
                           minLineLength=min_line_length, maxLineGap=max_line_gap)

    res = []

    if lines is None:
        return res
    lines = filter_too_similar_line(list(filter(hough_line_filter, lines)))
    lines_len = len(lines)

    for i in range(lines_len):
        line = lines[i]
        lx1, ly1, lx2, ly2 = line[0]
        for j in range(i + 1, lines_len):
            other = lines[j]
            rx1, ry1, rx2, ry2 = other[0]
            if not deviation_too_much(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):
                ltx, rbx = min_max(lx1, lx2, rx1, rx2)
                lty, rby = min_max(ly1, ly2, ry1, ry2)
                if (rbx - ltx) > 28 and (rby - lty) > 28:
                    res.append((ltx, lty, rbx, rby))

    return np_base_contour_filter(res)


class LineBaseRegionProposal:
    def __init__(self, hough_setting, sliding_window_setting):
        self.hough_setting = hough_setting
        self.sliding_window_setting = sliding_window_setting

    def __call__(self, rgb_matrix):
        '''
        :param rgb_matrix: (N, H, W, C)
        :return: LongTensor rois (N, 4), LongTensor roi_indices (N, )
        '''
        rgb_matrix = rgb_matrix.numpy()
        image_num = rgb_matrix.shape[0]
        rois = []
        roi_indices = []
        for i in range(image_num):
            single_matrix = rgb_matrix[i]
            res = houghline_base_proposal(single_matrix, **self.hough_setting)
            if len(res) == 0:
                res = sliding_window(single_matrix, **self.sliding_window_setting)
            rois.append(torch.tensor(res, dtype=torch.int))
            roi_indices.append(torch.full((len(res),), i, dtype=torch.int))

        return torch.cat(rois), torch.cat(roi_indices)


def sliding_window(rgb_matrix, x_start=36, y_start=36, step=32,
                   lengths=(128, 256, 384), scales=((1, 0.5), (1, 1), (1, 2))):
    h, w = rgb_matrix.shape[:2]
    res = []
    for length in lengths:
        for scale in scales:
            w_scale, h_scale = scale
            window_w = round(w_scale * length)
            window_h = round(h_scale * length)
            for y in range(y_start, h - window_h - y_start, step):
                for x in range(x_start, w - window_w - x_start, step):
                   res.append((x, y, x + window_w, y + window_h))

    print(len(res))
    if len(res) > 2000:
        return random.sample(res, 2000)
    return res

# def contour_filter(rect, width_minimum, width_maximum, height_minimum, height_maximum):
#     print(type(width_minimum < rect[2] < width_maximum and height_minimum < rect[3] < height_maximum), type(True))
#     return False


def is_surrounded_by(lhs, rhs):
    return lhs[0] > rhs[0] and lhs[1] > rhs[1] and lhs[2] < rhs[2] and lhs[3] < rhs[3]


def is_tightly_surrounded_by(lhs, rhs, threshold=14):
    return threshold >= (lhs[0] - rhs[0]) >= 0 and threshold >= (lhs[1] - rhs[1]) >= 0 \
           and threshold >= (rhs[2] - lhs[2]) >= 0 and threshold >= (rhs[3] - lhs[3]) >= 0


def np_base_contour_filter(contours, threshold=14):
    res = []
    contours = np.array(contours)
    contours_num = contours.shape[0]
    for i in range(contours_num):
        for_check = contours[i]
        indices = np.arange(contours_num) != i
        left = contours[indices]
        r1 = for_check[0] - left[:, 0]
        b1 = (threshold >= r1) & (r1 >= 0)
        r2 = for_check[1] - left[:, 1]
        b2 = (threshold >= r2) & (r2 >= 0)
        r3 = left[:, 2] - for_check[2]
        b3 = (threshold >= r3) & (r3 >= 0)
        r4 = left[:, 3] - for_check[3]
        b4 = (threshold >= r4) & (r4 >= 0)
        is_surrounded = (b1 & b2 & b3 & b4).any()
        if not is_surrounded:
            res.append(for_check)
    return res


def contour_filter(contours, filter_func=is_surrounded_by):
    res = []
    for box in contours:
        is_surrounded = False
        for other in contours:
            if (box is not other) and filter_func(box, other):
                is_surrounded = True
                break
        if not is_surrounded:
            res.append(box)

    return res


def contour_proposal(rgb_matrix, width_threshold=28, height_threshold=28):
    """
    By default, we will use the findContour method to create region proposal,
    however, if we can't find any approximate proposal, we will use sliding window 
    :param rgb_matrix: (H, W, C)
    :param width_threshold: 
    :param height_threshold: 
    :return: list of proposal region, each region is a tuple (ltx, lty, rbx, rby)
    """
    gray = cv.cvtColor(rgb_matrix, cv.COLOR_RGB2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    new_image_matrix, contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    height, width = new_image_matrix.shape[:2]
    width_maximum = width - 10
    height_maximum = height - 10

    res = []
    for p in map(cv.boundingRect, contours):
        x, y, w, h = p
        if width_threshold < w < width_maximum and height_threshold < h < height_maximum:
            res.append((x, y, x + w, y + h))

    return contour_filter(res)


def visualize(rgb_matrix, rectangles):
    image = cv.cvtColor(rgb_matrix, cv.COLOR_RGB2BGR)
    for rectangle in rectangles:
        ltx, lty, rbx, rby = rectangle
        cv.rectangle(image, (ltx, lty), (rbx, rby), (255, 0, 0), 2)
    cv.imshow("", image)


def auto_canny(gray_image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(gray_image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(gray_image, lower, upper)

    # return the edged image
    return edged


def deviation_too_much(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):
    if approximately_vertical(lx1, ly1, lx2, ly2) and approximately_vertical(rx1, ry1, rx2, ry2):
        ratio = overlap_ratio(ly1, ly2, ry1, ry2)
        if ratio < 0.7:
            return True

    if approximately_horizontal(lx1, ly1, lx2, ly2) and approximately_horizontal(rx1, ry1, rx2, ry2):
        ratio = overlap_ratio(lx1, lx2, rx1, rx2)
        if ratio < 0.7:
            return True

    return False


def approximately_horizontal_vertical(x1, y1, x2, y2):
    if abs(x1 - x2) <= 2.:
        return True
    k = abs((y1 - y2) / (x1 - x2))
    if k > 5.671 or k < 0.176:
        return True
    return False


def approximately_vertical(x1, y1, x2, y2):
    if abs(x1 - x2) <= 2.:
        return True
    k = abs((y1 - y2) / (x1 - x2))
    if k > 5.671:
        return True
    return False


def approximately_horizontal(x1, y1, x2, y2):
    if abs(x1 - x2) <= 2.:
        return False
    k = abs((y1 - y2) / (x1 - x2))
    if k < 0.176:
        return True
    return False


def hough_line_filter(line):
    x1, y1, x2, y2 = line[0]
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return approximately_horizontal_vertical(x1, y1, x2, y2)


def min_max(*seq):
    return min(seq), max(seq)


def test():
    dataloader = build_testloader(table_only=False)
    region_proposal = LineBaseRegionProposal({}, {})
    for tensor, origin_image, gt_box, gt_label in dataloader:
        rois, roi_indices = region_proposal(origin_image)
        image_for_show = origin_image[0].numpy()
        # print(rois.size(0))

        for i in range(rois.size(0)):
            roi = rois[i]
            cv.rectangle(image_for_show, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 1)
        cv.imshow("", image_for_show)
        cv.waitKey(0)

def np_test():
    a = np.random.rand(4)
    b = np.random.rand(5, 4)
    r1 = a[0] > b[:, 0]
    r2 = a[1] > b[:, 1]
    r3 = a[2] < b[:, 2]
    r4 = a[3] < b[:, 3]
    print(r1 & r2 & r3 & r4)
    print((r1 & r2 & r3 & r4).any())
    a = np.random.rand(4)
    print(is_surrounded_by(a, a))
    # print(c)


if __name__ == '__main__':
    test()
    # np_test()
    # a = (100, 100, 300, 100)
    # b = (70, 501, 88, 502)
    # print(overlap_ratio(a[0], a[2], b[0], b[2]))
    # for full_filename in os.listdir(ICDAR_2017_TEST + "image/"):
    #     image = cv.imread(os.path.join(ICDAR_2017_TEST + "image/", full_filename), cv.IMREAD_COLOR)
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #     proposal = LineBaseRegionProposal({}, {})
    #     h, w, c = image.shape
    #     windows, _ = proposal(torch.tensor(image.reshape(1, h, w, c)))
        # windows = houghline_base_proposal(image)
        # visualize(image, windows)
        # print(len(windows))
        # for window in windows:
        #     # copy = image.copy()
        #     cv.rectangle(image, tuple(window[:2]), tuple(window[2:]), (0, 0, 255), 1)
        #     # cv.imshow("", copy)
        #     # time.sleep(0.4)
        #     # cv.waitKey(1)
        #
        # cv.imshow("", image)
        # cv.waitKey(0)
        # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # # edges = cv.Canny(gray, 50, 150, apertureSize=3)
        # edges = auto_canny(gray)
        # rho = 1
        # theta = np.pi / 180
        # threshold = 30
        # min_line_length = 50
        # max_line_gap = 1
        # lines = cv.HoughLinesP(edges, rho, theta, threshold,
        #                        minLineLength=min_line_length, maxLineGap=max_line_gap)
        #
        # lines = list(filter(hough_line_filter, lines))
        # print(len(lines))
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv.imshow("", image)
        # cv.waitKey(0)

    # image = cv.imread(os.path.join(ICDAR_2017_TEST + "image/", "POD_1612.bmp"), cv.IMREAD_COLOR)
    # windows = houghline_base_proposal(image)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # # edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # edges = auto_canny(gray)
    # rho = 1
    # theta = np.pi / 180
    # threshold = 30
    # min_line_length = 50
    # max_line_gap = 2
    # lines = cv.HoughLinesP(edges, rho, theta, threshold,
    #                        minLineLength=min_line_length, maxLineGap=max_line_gap)
    #
    # lines = list(filter(hough_line_filter, lines))
    # print(len(lines))
    # for line in lines:
    #     # copy = image.copy()
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #     # print(x1, y1, x2, y2)
    #     # cv.imshow("", copy)
    #     # time.sleep(0.4)
    #     # cv.waitKey(1)
    # cv.imshow("", image)
    # cv.waitKey(0)

    # print(len(windows))
    # for window in windows:
    #     # copy = image.copy()
    #     cv.rectangle(image, tuple(window[:2]), tuple(window[2:]), (0, 0, 255), 1)
    #     # cv.imshow("", copy)
    #     # time.sleep(0.4)
    #     # cv.waitKey(1)
    # cv.imshow("", image)
    # cv.waitKey(0)


