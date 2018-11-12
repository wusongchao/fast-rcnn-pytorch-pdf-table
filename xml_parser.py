from path_constant import *

import os
import xml.etree.ElementTree as ET
import struct
import binascii
import cv2 as cv
import math


if __name__ == '__main__':
    image_path = MARMOT_CHINESE + "image/"
    label_path = MARMOT_CHINESE + "label/"
    for filename in os.listdir(label_path):
        scaling_x = 1.0
        scaling_y = 1.0
        # filename = "9_40.xml"
        prefix, _ = os.path.splitext(filename)
        image = cv.imread(os.path.join(image_path, prefix) + ".bmp")
        h, w = image.shape[:2]
        print(filename)
        print(image.shape)
        tree = ET.ElementTree(file=os.path.join(label_path, filename))
        root = tree.getroot()
        print(root.tag)
        if root.tag == "Page":
            lbx, lby, rtx, rty = [struct.unpack(">d", binascii.unhexlify(hex_str)) for hex_str in
                                  root.get("CropBox").split(" ")]
            lbx = lbx[0]
            lby = lby[0]
            rtx = rtx[0]
            rty = rty[0]

            scaling_x = w / (rtx - lbx)
            scaling_y = h / (lby - rty)


            ltx = round(lbx * scaling_x)
            lty = round(rty * scaling_y)
            rbx = round(rtx * scaling_x)
            rby = round(lby * scaling_y)
            # print((ltx, lty), (rbx, rby))
            # cv.rectangle(image, (ltx, lty), (rbx, rby), (255, 0, 0), 1)

        for elem in tree.iter(tag='Composite'):
            if elem.get("Label") == "TableBody":
                lbx, lby, rtx, rty = [struct.unpack(">d", binascii.unhexlify(hex_str)) for hex_str in elem.attrib["BBox"].split(" ")]
                lbx = lbx[0]
                lby = lby[0]
                rtx = rtx[0]
                rty = rty[0]

                print(scaling_x, scaling_y)
                ltx = round(lbx * scaling_x)
                lty = round(rty * scaling_y)
                rbx = round(rtx * scaling_x)
                rby = round(lby * scaling_y)
                print((ltx, lty), (rbx, rby))
                # print((int(x1[0]), int(y1[0])), (int(x2[0]), int(y2[0])))
                cv.rectangle(image, (ltx, h - lty), (rbx, h - rby), (255, 0, 0), 1)

        cv.imshow("test", image)
        cv.waitKey(0)
        # break