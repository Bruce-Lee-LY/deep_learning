# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: Caltech_256

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import cv2
import numpy as np
import random
from datetime import datetime
from tensorflow.python.keras.utils import to_categorical

CALTECH_256_PATH = "dataset/Caltech/256_ObjectCategories/"
CALTECH_256_LABEL = "dataset/Caltech/Caltech_256_label.txt"
CALTECH_256_TRAIN = "dataset/Caltech/Caltech_256_train.txt"
CALTECH_256_VAL = "dataset/Caltech/Caltech_256_val.txt"
CALTECH_256_TEST = "dataset/Caltech/Caltech_256_test.txt"


class Sample:
    def __init__(self, image_name, label):
        self.image_name = image_name
        self.label = label


class Batch:
    def __init__(self, images, labels):
        self.images = np.stack(images, axis=0)
        self.labels = labels


class Caltech_256:
    def __init__(
            self,
            width,
            height,
            channel,
            train_batch_size,
            val_batch_size,
            test_batch_size):
        self.caltech_path = CALTECH_256_PATH
        self.caltech_label = CALTECH_256_LABEL
        self.caltech_train = CALTECH_256_TRAIN
        self.caltech_val = CALTECH_256_VAL
        self.caltech_test = CALTECH_256_TEST
        self.caltech_dirs = os.listdir(self.caltech_path)
        self.caltech_dirs.sort()
        self.caltech_classes = len(self.caltech_dirs)
        print("Caltech_256: num_classes =", self.caltech_classes)
        self.caltech_files = [[] for _ in range(257)]
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.cv2_size = (width, height)
        if channel == 1:
            self.cv2_fmt = cv2.IMREAD_GRAYSCALE
        else:
            self.cv2_fmt = cv2.IMREAD_COLOR

        self.get_labels()
        self.get_samples()

    def get_labels(self):
        with open(self.caltech_label, 'w', encoding='utf-8') as fw:
            for dir in self.caltech_dirs:
                fw.write(dir + '\n')

    def get_samples(self):
        it = 0
        for dir in self.caltech_dirs:
            for _, _, filename in os.walk(
                    os.path.join(self.caltech_path, dir)):
                for i in filename:
                    self.caltech_files[it].append(os.path.join(
                        os.path.join(self.caltech_path, dir), i))
            it += 1

        with open(self.caltech_test, 'w', encoding='utf-8') as fwt:
            for i in range(len(self.caltech_files)):
                for j in range(1):
                    image_name = os.path.join(
                        self.caltech_path, self.caltech_files[i][j])
                    label = str(i)
                    fwt.write(image_name + ' ' + label + '\n')
                    self.test_samples.append(Sample(image_name, label))
        random.shuffle(self.test_samples)
        self.test_num = len(self.test_samples)
        print("Caltech_256: test_num = {}, test_batch_size = {}, test_batch = {}".format(
            self.test_num, self.test_batch_size, self.test_num // self.test_batch_size))
        assert(self.test_batch_size < self.test_num)

        with open(self.caltech_val, 'w', encoding='utf-8') as fwv:
            for i in range(len(self.caltech_files)):
                for j in range(1, 10):
                    image_name = os.path.join(
                        self.caltech_path, self.caltech_files[i][j])
                    label = str(i)
                    fwv.write(image_name + ' ' + label + '\n')
                    self.val_samples.append(Sample(image_name, label))
        random.shuffle(self.val_samples)
        self.val_num = len(self.val_samples)
        print("Caltech_256: val_num = {}, val_batch_size = {}, val_batch = {}".format(
            self.val_num, self.val_batch_size, self.val_num // self.val_batch_size))
        assert(self.val_batch_size < self.val_num)

        with open(self.caltech_train, 'w', encoding='utf-8') as fwt:
            for i in range(len(self.caltech_files)):
                for j in range(10, len(self.caltech_files[i])):
                    image_name = os.path.join(
                        self.caltech_path, self.caltech_files[i][j])
                    label = str(i)
                    fwt.write(image_name + ' ' + label + '\n')
                    self.train_samples.append(Sample(image_name, label))
        random.shuffle(self.train_samples)
        self.train_num = len(self.train_samples)
        print("Caltech_256: train_num = {}, train_batch_size = {}, train_batch = {}".format(
            self.train_num, self.train_batch_size, self.train_num // self.train_batch_size))
        assert(self.train_batch_size < self.train_num)

    def create_image(self, image_name):
        image = cv2.imread(image_name, self.cv2_fmt)
        image = cv2.resize(image, self.cv2_size, interpolation=cv2.INTER_AREA)
        mean, std = cv2.meanStdDev(image)
        mean, std = mean.astype(np.float32), std.astype(np.float32)
        image = image.astype(np.float32)
        image = (image - np.squeeze(mean)) / \
            (np.squeeze(std) + np.finfo(np.float32).eps)
        return image

    def get_train_batch(self):
        batch_range = range(
            self.train_idx,
            self.train_idx +
            self.train_batch_size)
        labels = [self.train_samples[i].label for i in batch_range]
        images = [
            self.create_image(
                self.train_samples[i].image_name) for i in batch_range]
        self.train_idx += self.train_batch_size
        return Batch(images, to_categorical(labels, self.caltech_classes))

    def get_val_batch(self):
        batch_range = range(self.val_idx, self.val_idx + self.val_batch_size)
        labels = [self.val_samples[i].label for i in batch_range]
        images = [
            self.create_image(
                self.val_samples[i].image_name) for i in batch_range]
        self.val_idx += self.val_batch_size
        return Batch(images, to_categorical(labels, self.caltech_classes))

    def get_test_batch(self):
        batch_range = range(
            self.test_idx,
            self.test_idx +
            self.test_batch_size)
        labels = [self.test_samples[i].label for i in batch_range]
        images = [
            self.create_image(
                self.test_samples[i].image_name) for i in batch_range]
        self.test_idx += self.test_batch_size
        return Batch(images, to_categorical(labels, self.caltech_classes))

    def can_get_train_batch(self):
        return self.train_idx + self.train_batch_size < self.train_num

    def can_get_val_batch(self):
        return self.val_idx + self.val_batch_size < self.val_num

    def can_get_test_batch(self):
        return self.test_idx + self.test_batch_size < self.test_num

    def reset_train_batch(self):
        self.train_idx = 0
        random.shuffle(self.train_samples)

    def reset_val_batch(self):
        self.val_idx = 0
        # random.shuffle(self.val_samples)

    def reset_test_batch(self):
        self.test_idx = 0
        random.shuffle(self.test_samples)


def main():
    caltech_start = datetime.now()
    caltech_256 = Caltech_256(224, 224, 3, 1000, 100, 1)
    i = 0
    while caltech_256.can_get_train_batch() and caltech_256.can_get_val_batch(
    ) and caltech_256.can_get_test_batch():
        batch_start = datetime.now()
        train_batch = caltech_256.get_train_batch()
        val_batch = caltech_256.get_val_batch()
        test_batch = caltech_256.get_test_batch()
        i += 1
        print("Batch[{}]: taken = {}".format(i, datetime.now() - batch_start))
    print(
        "Caltech_256: total batch = {}, total taken = {}".format(
            i,
            datetime.now() -
            caltech_start))


if __name__ == "__main__":
    main()
