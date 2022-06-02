# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: VOC to YOLOLabel

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import xml.etree.ElementTree as ET

VOC_DATASET_PATH = "dataset/VOCdevkit/"

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test'),
        ('2012', 'train'), ('2012', 'val')]
classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id):
    in_file = open(
        VOC_DATASET_PATH + 'VOC%s/Annotations/%s.xml' %
        (year, image_id))
    out_file = open(
        VOC_DATASET_PATH + 'VOC%s/labels/%s.txt' %
        (year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (
            float(
                xmlbox.find('xmin').text), float(
                xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(
                    xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


def VOC_to_YOLOLabel():
    for year, image_set in sets:
        if not os.path.exists(VOC_DATASET_PATH + 'VOC%s/labels/' % (year)):
            os.makedirs(VOC_DATASET_PATH + 'VOC%s/labels/' % (year))
        image_ids = open(
            VOC_DATASET_PATH + 'VOC%s/ImageSets/Main/%s.txt' %
            (year, image_set)).read().strip().split()
        list_file = open(
            VOC_DATASET_PATH + 'VOC%s/%s_%s.txt' %
            (year, year, image_set), 'w')
        for image_id in image_ids:
            list_file.write(
                VOC_DATASET_PATH + 'VOC%s/JPEGImages/%s.jpg\n' %
                (year, image_id))
            convert_annotation(year, image_id)
        list_file.close()


def main():
    VOC_to_YOLOLabel()


if __name__ == "__main__":
    main()
