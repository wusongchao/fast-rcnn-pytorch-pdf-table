import xml.etree.ElementTree as ET
import cv2 as cv
from path_constant import *
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import struct
import binascii
import random
import utils_func


CLASS_BACKGROUND = 0
CLASS_TABLE = 1
CLASS_FIGURE = 2


def str_tuple_to_int_point(str_tuple):
    x, y = str_tuple.split(",")
    return round(float(x)), round(float(y))


def extract_bounding_box_node(bounding_box_node, page_height):
    # (x1, y1) lb
    # (x2, y2) rt
    # lb origin is (0, 0)
    x1 = bounding_box_node.get("x1")
    y1 = bounding_box_node.get("y1")
    x2 = bounding_box_node.get("x2")
    y2 = bounding_box_node.get("y2")
    # "fix" the coordinate
    return int(x1)-3, page_height - int(y2), int(x2)+7, page_height - int(y1)+3
    # return int(x1), page_height - int(y1), int(x2), page_height - int(y2)


def extract_coords_node(coords_node):
    lt, _, _, rb = coords_node.get("points").split(' ')
    ltx, lty = lt.split(',')
    rbx, rby = rb.split(',')
    return int(ltx), int(lty), int(rbx), int(rby)


class ICDAR2013(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.filename_list = os.listdir(self.image_dir)
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        image_name = self.filename_list[index]
        filename, _ = os.path.splitext(image_name)
        xml_name, image_page_num = filename.split("_")
        image_page_num = int(image_page_num)
        tables = ET.ElementTree(file=os.path.join(self.label_dir, xml_name + ".xml")).getroot().findall("table")
        bounding_boxes = []
        boxes_labels = []
        image_matrix = cv.imread(os.path.join(self.image_dir, image_name), cv.IMREAD_COLOR)

        for table in tables:
            regions = table.findall("region")
            for region in regions:
                table_page_num = int(region.get("page"))
                if image_page_num == table_page_num:
                    bbox_node = region.find("bounding-box")
                    bounding_boxes.append(extract_bounding_box_node(bbox_node, image_matrix.shape[0]))
                    boxes_labels.append(CLASS_TABLE)

        image_matrix = cv.cvtColor(image_matrix, cv.COLOR_BGR2RGB)
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.int)
        boxes_labels = torch.tensor(boxes_labels, dtype=torch.long)

        sample = image_matrix, bounding_boxes, boxes_labels

        if self.transform:
            return self.transform(sample)
        return sample


class ICDAR2017(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, table_only=False):
        self.image_dir = image_dir
        self.filename_list = os.listdir(self.image_dir)
        self.label_dir = label_dir
        self.transform = transform
        self.table_only = table_only

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        image_name = self.filename_list[index]
        filename, _ = os.path.splitext(image_name)
        xml_name = filename
        root = ET.ElementTree(file=os.path.join(self.label_dir, xml_name + ".xml")).getroot()
        bounding_boxes = []
        boxes_labels = []
        image_matrix = cv.imread(os.path.join(self.image_dir, image_name), cv.IMREAD_COLOR)

        tables = root.findall("tableRegion")
        for table in tables:
            coords_node = table.find("Coords")
            bounding_boxes.append(extract_coords_node(coords_node))
            boxes_labels.append(CLASS_TABLE)

        if not self.table_only:
            figures = root.findall("figureRegion")
            for figure in figures:
                coords_node = figure.find("Coords")
                bounding_boxes.append(extract_coords_node(coords_node))
                boxes_labels.append(CLASS_FIGURE)

        image_matrix = cv.cvtColor(image_matrix, cv.COLOR_BGR2RGB)
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.int)
        boxes_labels = torch.tensor(boxes_labels, dtype=torch.long)

        sample = image_matrix, bounding_boxes, boxes_labels

        if self.transform:
            return self.transform(sample)
        return sample


def calculate_scaling_factor(root, w, h):
    lbx, lby, rtx, rty = [struct.unpack(">d", binascii.unhexlify(hex_str)) for hex_str in
                          root.get("CropBox").split(" ")]
    lbx = lbx[0]
    lby = lby[0]
    rtx = rtx[0]
    rty = rty[0]

    scaling_x = w / (rtx - lbx)
    scaling_y = h / (lby - rty)

    return scaling_x, scaling_y


def extract_marmot_bbox(node, scaling_x, scaling_y, page_height):
    ltx, lty, rbx, rby = [struct.unpack(">d", binascii.unhexlify(hex_str)) for hex_str in node.get("BBox").split(" ")]
    ltx = ltx[0]
    lty = lty[0]
    rbx = rbx[0]
    rby = rby[0]

    ltx = round(ltx * scaling_x)
    lty = round(lty * scaling_y)
    rbx = round(rbx * scaling_x)
    rby = round(rby * scaling_y)
    # print(ltx, lty, rbx, rby)

    return ltx, page_height - lty, rbx, page_height - rby


class MARMOT(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, table_only=False):
        self.image_dir = image_dir
        self.filename_list = os.listdir(self.image_dir)
        self.label_dir = label_dir
        self.transform = transform
        self.table_only = table_only

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        image_name = self.filename_list[index]
        filename, _ = os.path.splitext(image_name)
        xml_name = filename

        bounding_boxes = []
        boxes_labels = []
        image_matrix = cv.imread(os.path.join(self.image_dir, image_name), cv.IMREAD_COLOR)
        h, w = image_matrix.shape[:2]

        tree = ET.ElementTree(file=os.path.join(self.label_dir, xml_name + ".xml"))
        root = tree.getroot()

        scaling_x = 1.0
        scaling_y = 1.0

        if root.tag == "Page":
            scaling_x, scaling_y = calculate_scaling_factor(root, w, h)

        for elem in tree.iter(tag='Composite'):
            label_str = elem.get("Label")
            if label_str == "TableBody":
                bounding_boxes.append(extract_marmot_bbox(elem, scaling_x, scaling_y, h))
                boxes_labels.append(CLASS_TABLE)
            elif not self.table_only and label_str == "Figure":
                bounding_boxes.append(extract_marmot_bbox(elem, scaling_x, scaling_y, h))
                boxes_labels.append(CLASS_FIGURE)

        image_matrix = cv.cvtColor(image_matrix, cv.COLOR_BGR2RGB)
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.int)
        boxes_labels = torch.tensor(boxes_labels, dtype=torch.long)

        sample = image_matrix, bounding_boxes, boxes_labels

        if self.transform:
            return self.transform(sample)
        return sample


# def restrict_image(image_matrix, max_size):
#     h, w = image_matrix.shape[1:]
#     maximum_dimension = max(h, w)
#     if maximum_dimension <= max_size:
#         return


class RestrictImageSize:
    def __init__(self, max_size=1000., min_size=600.):
        self.max_size = max_size
        self.min_size = min_size

    def __call__(self, sample):
        image_matrix, bounding_boxes, boxes_labels = sample
        h, w = image_matrix.shape[:2]
        maximum_dimension, minimum_dimension = (h, w) if h > w else (w, h)
        if maximum_dimension <= self.max_size and minimum_dimension >= self.min_size:
            return image_matrix, bounding_boxes, boxes_labels
        elif maximum_dimension > self.max_size:
            scale_factor = self.max_size / maximum_dimension
            image_matrix = cv.resize(image_matrix, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
            bounding_boxes = utils_func.resize_bbox(bounding_boxes, scale_factor)
            return image_matrix, bounding_boxes, boxes_labels
        elif minimum_dimension < self.min_size:
            scale_factor = self.min_size / minimum_dimension
            image_matrix = cv.resize(image_matrix, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
            bounding_boxes = utils_func.resize_bbox(bounding_boxes, scale_factor)
            return image_matrix, bounding_boxes, boxes_labels


def flip_bounding_boxes(bounding_boxes, page_width):
    res = []
    for box in bounding_boxes:
        ltx, lty, rbx, rby = box
        flip_rbx = page_width - ltx
        flip_ltx = page_width - rbx
        res.append((flip_ltx, lty, flip_rbx, rby))
    return torch.tensor(res, dtype=torch.int)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_matrix, bounding_boxes, boxes_labels = sample
        if random.random() < self.p:
            return cv.flip(image_matrix, 1), flip_bounding_boxes(bounding_boxes, image_matrix.shape[1]), boxes_labels
        return image_matrix, bounding_boxes, boxes_labels


class ToTensor:
    def __init__(self):
        self.fn = transforms.ToTensor()
        transforms.ToTensor()

    def __call__(self, sample):
        image_matrix, bounding_boxes, boxes_labels = sample
        return self.fn(image_matrix), image_matrix, bounding_boxes, boxes_labels


class Normalize:
    def __init__(self, mean, std):
        self.fn = transforms.Normalize(mean, std)

    def __call__(self, sample):
        image_tensor, image_matrix, bounding_boxes, boxes_labels = sample
        return self.fn(image_tensor), image_matrix, bounding_boxes, boxes_labels


def build_dataloader(table_only=False):
    transforms_action = transforms.Compose([
        RestrictImageSize(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    icdar_2013_train = ICDAR2013(ICDAR_TRAIN + "image/", ICDAR_TRAIN + "label/", transform=transforms_action)
    icdar_2013_competition = ICDAR2013(ICDAR_COMPETITION + "image/", ICDAR_COMPETITION + "label/", transform=transforms_action)
    if table_only:
        icdar_2017_train = ICDAR2017(ICDAR_2017_TRAIN_TABLEONLY + "image/", ICDAR_2017_TRAIN_TABLEONLY + "label/", transform=transforms_action, table_only=table_only)
        marmot_chinese = MARMOT(MARMOT_CHINESE_TABLEONLY + "image/", MARMOT_CHINESE_TABLEONLY + "label/", transform=transforms_action, table_only=table_only)
        marmot_english = MARMOT(MARMOT_ENGLISH_TABLEONLY + "image/", MARMOT_ENGLISH_TABLEONLY + "label/", transform=transforms_action, table_only=table_only)
    else:
        icdar_2017_train = ICDAR2017(ICDAR_2017_TRAIN + "image/", ICDAR_2017_TRAIN + "label/", transform=transforms_action)
        marmot_chinese = MARMOT(MARMOT_CHINESE + "image/", MARMOT_CHINESE + "label/", transform=transforms_action)
        marmot_english = MARMOT(MARMOT_ENGLISH + "image/", MARMOT_ENGLISH + "label/", transform=transforms_action)
    trainset = ConcatDataset([icdar_2013_train, icdar_2013_competition, icdar_2017_train, marmot_chinese, marmot_english])
    # trainset = ConcatDataset([icdar_2017_train])
    # only support batch_size=1, since data is a variable-length sequence
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    return trainloader


def build_testloader(table_only=False):
    transforms_action = transforms.Compose([
        RestrictImageSize(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # icdar_2013_competition = ICDAR2013(ICDAR_COMPETITION + "image/", ICDAR_COMPETITION + "label/",
    #                                    transform=transforms_action)
    if table_only:
        icdar_2017_competition = ICDAR2017(ICDAR_2017_TEST_TABLEONLY + "image/", ICDAR_2017_TEST_TABLEONLY + "label/", transform=transforms_action, table_only=table_only)
    else:
        icdar_2017_competition = ICDAR2017(ICDAR_2017_TEST + "image/", ICDAR_2017_TEST + "label/",
                                           transform=transforms_action)
    testset = ConcatDataset([icdar_2017_competition])
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    return testloader

if __name__ == '__main__':
    transforms_action = transforms.Compose([
        RestrictImageSize(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    icdar_2013_train = ICDAR2013(ICDAR_TRAIN + "image/", ICDAR_TRAIN + "label/", transform=transforms_action)
    icdar_2013_competition = ICDAR2013(ICDAR_COMPETITION + "image/", ICDAR_COMPETITION + "label/", transform=transforms_action)
    icdar_2017_train = ICDAR2017(ICDAR_2017_TRAIN + "image/", ICDAR_2017_TRAIN + "label/", transform=transforms_action)
    marmot_chinese = MARMOT(MARMOT_CHINESE + "image/", MARMOT_CHINESE + "label/", transform=transforms_action)
    marmot_english = MARMOT(MARMOT_ENGLISH + "image/", MARMOT_ENGLISH + "label/", transform=transforms_action)
    trainset = ConcatDataset([marmot_chinese, marmot_english])
    # only support batch_size=1, since data is a variable-length sequence
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

    for i, data in enumerate(trainloader):
        # print(i)
        tensor, origin_image, boxes, labels = data
        # print(tensor.size(), origin_image.numpy(), boxes.size(), labels.size())
        # break
        height = origin_image.shape[0]
        origin_image = origin_image.numpy()[0]
        print(origin_image.shape)
        boxes = boxes[0]
        label = labels[0]
        for i in range(boxes.size(0)):
            box = boxes[i]
            cv.rectangle(origin_image, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 1)
            cv.putText(origin_image, str(label[i].item()), (box[0].item(), box[1].item()),
                       cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv.imshow(str(i), origin_image)
        cv.waitKey(0)
        # if i == 3:
        #     break
        # break
    # image_path = ICDAR_2017_TRAIN + "image/"
    # label_path = ICDAR_2017_TRAIN + "label/"
    # for full_filename in os.listdir(label_path):
    #     prefix, _ = os.path.splitext(full_filename)
    #     tree = ET.ElementTree(file=os.path.join(label_path, full_filename))
    #     root = tree.getroot()
    #     tables = root.findall("tableRegion")
    #     figures = root.findall("figureRegion")
    #     image_matrix = cv.imread(os.path.join(image_path, prefix + ".bmp"))
    #     for table in tables:
    #         lt, _, _, rb = table.find("Coords").get("points").split(" ")
    #         cv.rectangle(image_matrix, str_tuple_to_int_point(lt), str_tuple_to_int_point(rb), (255, 0, 0), 1)
    #     for figure in figures:
    #         lt, _, _, rb = figure.find("Coords").get("points").split(" ")
    #         cv.rectangle(image_matrix, str_tuple_to_int_point(lt), str_tuple_to_int_point(rb), (255, 0, 0), 1)
    #     cv.imshow("im", image_matrix)
    #     cv.waitKey(0)
    #     break

    # image_path = ICDAR_TRAIN + "image/"
    # label_path = ICDAR_TRAIN + "label/"
    # for full_image_name in os.listdir(image_path):
    # #     prefix, _ = os.path.splitext(full_image_name)
    # #     split = prefix.split("_")
    # #     page_num = split[-1]
    # #     label_filename = split[0]
    # #     tree = ET.ElementTree(file=os.path.join(label_path, label_path + ".xml"))
    # #     tables = tree.getroot().findall("table")
    # #     for table in tables
    # #     image_matrix = cv.imread(os.path.join(image_path, full_image_name))
    # #     for table in tables:
    # #         lt, _, _, rb = table.find("Coords").get("points").split(" ")
    # #         cv.rectangle(image_matrix, str_tuple_to_int_point(lt), str_tuple_to_int_point(rb), (255, 0, 0), 1)
    # #     cv.imshow("im", image_matrix)
    # #     cv.waitKey(0)
    #     full_image_name = "eu-003_1.png"
    #     image = cv.imread(os.path.join(image_path, full_image_name))
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #     plt.imshow(image)
    #     plt.show()
    #     break
