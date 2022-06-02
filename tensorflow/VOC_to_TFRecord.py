# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: VOC to TFRecord

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import sys
import random
import tensorflow as tf
from PIL import Image
import xml.etree.ElementTree as ET

VOC_DATASET_PATH = "dataset/VOCdevkit/"
DIRECTORY_ANNOTATIONS = './Annotations/'
DIRECTORY_IMAGES = './JPEGImages/'
RANDOM_SEED = 666
SAMPLES_PER_FILES = 2000

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'trainval'),
        ('2012', 'train'), ('2012', 'val'), ('2012', 'trainval')]

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def int64_feature(values):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def process_image(directory, name):
    # Read the image file.
    filename = os.path.join(directory, DIRECTORY_IMAGES, name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    with Image.open(filename) as img:
        print(img.size)
    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append(
            (float(
                bbox.find('ymin').text) /
                shape[0],
                float(
                bbox.find('xmin').text) /
                shape[1],
                float(
                bbox.find('ymax').text) /
                shape[0],
                float(
                bbox.find('xmax').text) /
                shape[1]))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def convert_to_example(
        image_data,
        shape,
        bboxes,
        labels,
        labels_text,
        difficult,
        truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = process_image(
        dataset_dir, name)
    print(shape, bboxes, labels, labels_text, difficult, truncated)
    example = convert_to_example(
        image_data,
        shape,
        bboxes,
        labels,
        labels_text,
        difficult,
        truncated)
    tfrecord_writer.write(example.SerializeToString())


def VOC_to_TFRecord(shuffling=False):
    for year, image_set in sets:
        output_dir = VOC_DATASET_PATH + \
            'VOC%s/TFRecord/%s/' % (year, image_set)
        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)

        split_file_path = os.path.join(
            VOC_DATASET_PATH,
            'VOC%s' %
            year,
            'ImageSets',
            'Main',
            '%s.txt' %
            image_set)
        print('>> ', split_file_path)
        with open(split_file_path) as f:
            filenames = f.readlines()

        if shuffling:
            random.seed(RANDOM_SEED)
            random.shuffle(filenames)
        # Process dataset files.
        i = 0
        fidx = 0
        dataset_dir = os.path.join(VOC_DATASET_PATH, 'VOC%s' % year)
        while i < len(filenames):
            # Open new TFRecord file.
            tf_filename = '%s/%s_%03d.tfrecord' % (output_dir, image_set, fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < len(filenames) and j < SAMPLES_PER_FILES:
                    sys.stdout.write(
                        '\r>> Converting image %d/%d' %
                        (i + 1, len(filenames)))
                    sys.stdout.flush()
                    filename = filenames[i].strip()
                    add_to_tfrecord(dataset_dir, filename, tfrecord_writer)
                    i += 1
                    j += 1
                fidx += 1
        print(
            '\n>> Finished converting the Pascal VOC%s %s dataset!' %
            (year, image_set))


def main():
    VOC_to_TFRecord()


if __name__ == "__main__":
    main()
