from path_constant import *
import os
import shutil
from wand.image import Image
from wand.color import Color
import chainercv.evaluations.eval_detection_voc
import xml.etree.ElementTree as ET


def clear(path):
    for full_filename in os.listdir(path):
        filename, ext_name = os.path.splitext(full_filename)
        # if ext_name == ".csv" or ext_name == ".xls" or ext_name == ".xlsx":
        if ext_name == ".pdf":
            os.remove(os.path.join(path, full_filename))


def move(src_path, dest_path):
    for full_filename in os.listdir(src_path):
        filename, ext_name = os.path.splitext(full_filename)
        # if ext_name == ".xml":
        #     # print("-".join(filename.split("-")[:-1]) + ".xml")
        #     new_full_filename = "-".join(filename.split("-")[:-1]) + ".xml"
        #     shutil.move(os.path.join(src_path, full_filename), os.path.join(dest_path, new_full_filename))
        #
        if ext_name == ".bmp":
            shutil.move(os.path.join(src_path, full_filename), os.path.join(dest_path, full_filename))


def move_pdf(src_path, dest_path):
    for full_filename in os.listdir(src_path):
        filename, ext_name = os.path.splitext(full_filename)
        if ext_name == ".pdf":
            shutil.copy(os.path.join(src_path, full_filename), os.path.join(dest_path, full_filename))


def pdf_to_png(src_path):
    for full_filename in os.listdir(src_path):
        filename, ext_name = os.path.splitext(full_filename)
        if ext_name == ".pdf":
            with Image(filename=os.path.join(src_path, full_filename)) as all_pages:
                for i, single_page in enumerate(all_pages.sequence, 1):
                    with Image(single_page).convert("png") as converted:
                        converted.background_color = Color("white")
                        converted.alpha_channel = "remove"
                        converted.save(filename=os.path.join(src_path, filename + "_%d" % i + ".png"))


def clean_necessary_png(label_path, image_path):
    has = image_path + "has/"
    for full_filename in os.listdir(label_path):
        prefix, _ = os.path.splitext(full_filename)
        tree = ET.ElementTree(file=os.path.join(label_path, full_filename))
        for table in tree.iter(tag="table"):
            region = table.find("region")
            page_num = int(region.get("page"))
            image_name = prefix + "_%d" % (page_num) + ".png"
            shutil.copy(os.path.join(image_path, image_name), os.path.join(has, image_name))


def filter_icdar_2017(label_path, image_path):
    for full_filename in os.listdir(label_path):
        prefix, _ = os.path.splitext(full_filename)
        tree = ET.ElementTree(file=os.path.join(label_path, full_filename))
        root = tree.getroot()
        if not root.findall("tableRegion") and not root.findall("figureRegion"):
            os.remove(os.path.join(label_path, full_filename))
            if os.path.exists(os.path.join(image_path, prefix + ".bmp")):
                os.remove(os.path.join(image_path, prefix + ".bmp"))
        # if not root.findall("tableRegion"):
        #     os.remove(os.path.join(label_path, full_filename))
        #     os.remove(os.path.join(image_path, prefix + ".bmp"))
        # for table in tree.iter(tag="table"):
        #     region = table.find("region")
        #     page_num = int(region.get("page"))
        #     image_name = prefix + "-%d" % (page_num - 1) + ".png"
        #     shutil.copy(os.path.join(image_path, image_name), os.path.join(has, image_name))


def filter_marmot(label_path, image_path):
    for full_filename in os.listdir(label_path):
        prefix, _ = os.path.splitext(full_filename)
        tree = ET.ElementTree(file=os.path.join(label_path, full_filename))
        has = False
        for elem in tree.iter(tag="Composite"):
            label_str = elem.get("Label")
            # if label_str == "TableBody" or label_str == "Figure":
            #     has = True
            if label_str == "TableBody":
                has = True

        if has == False:
            os.remove(os.path.join(label_path, full_filename))
            os.remove(os.path.join(image_path, prefix + ".bmp"))


# def change_icdar_image_name(image_path, new_path):
#     for full_filename in os.listdir(image_path):
#         prefix, _ = os.path.splitext(full_filename)
#         split = prefix.split('_')
#         page_num = split[-1]
#         print(full_filename)
#         print(page_num)
#         prefix = split[0] + '_' + str((int(page_num) + 1))
#         # os.rename(os.path.join(image_path, full_filename), os.path.join(new_path, prefix+".png"))


if __name__ == '__main__':

    # label_path = ICDAR_2017_TEST + "label/"
    # image_path = ICDAR_2017_TEST + "image/"
    # # new_path = ICDAR_COMPETITION + "has/"
    # # clean_necessary_png(label_path, image_path)
    # filter_icdar_2017(label_path, image_path)
    # change_icdar_image_name(image_path, new_path)
    # move_pdf("H:\dl\icdar dataset\eu-dataset", ICDAR_TRAIN + "image/")
    # move_pdf("H:\dl\icdar dataset\\us-gov-dataset", ICDAR_TRAIN + "image/")
    # move_pdf("H:\dl\icdar dataset\competition-dataset-eu", ICDAR_COMPETITION + "image/")
    # move_pdf("H:\dl\icdar dataset\competition-dataset-us", ICDAR_COMPETITION + "image/")
    # pdf_to_png(ICDAR_TRAIN + "image/")
    # pdf_to_png(ICDAR_COMPETITION + "image/")
    # clean_necessary_png(ICDAR_TRAIN + "label/", ICDAR_TRAIN + "image/")
    # clean_necessary_png(ICDAR_COMPETITION + "label/", ICDAR_COMPETITION + "image/")
    # clear(ICDAR_TRAIN + "image/")
    # clear(ICDAR_COMPETITION + "image/")
    # filter_icdar_2017(ICDAR_2017_TRAIN_TABLEONLY + "label/", ICDAR_2017_TRAIN_TABLEONLY + "image/")
    # filter_icdar_2017(ICDAR_2017_TEST_TABLEONLY + "label/", ICDAR_2017_TEST_TABLEONLY + "image/")
    # filter_marmot(MARMOT_CHINESE_TABLEONLY + "label/", MARMOT_CHINESE_TABLEONLY + "image/")
    # filter_marmot(MARMOT_ENGLISH_TABLEONLY + "label/", MARMOT_ENGLISH_TABLEONLY + "image/")
    filter_icdar_2017(ICDAR_2017_TRAIN + "label/", ICDAR_2017_TRAIN + "image/")