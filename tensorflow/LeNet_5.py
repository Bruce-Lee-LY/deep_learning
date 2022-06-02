# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: LeNet_5

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import sys
from datetime import datetime
import json
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

MNIST_DATASET_PATH = "dataset/MNIST_data/"
LeNet_5_MODEL_PATH = "model/LeNet_5/"
LeNet_5_MODEL_NAME = "LeNet_5_train_step"
LeNet_5_MODEL_RECORD = "LeNet_5_model_record.json"
LeNet_5_TRAIN_RECORD = "LeNet_5_train_record.png"
LeNet_5_TENSORBOARD_PATH = "tensorboard/LeNet_5/"


class LeNet_5:
    '''
    layer       input       kernel      feature_map     neuron_num      train_params        connect_num
    conv1       32*32       5*5*6       28*28*6         28*28*6         (5*5+1)*6           (5*5+1)*6*28*28
    mpool1      28*28*6     2*2*6       14*14*6         -               -                   -
    conv2       14*14*6     5*5*16      10*10*16        10*10*16    (5*5*3+1)*6+(5*5*4+1)*9+(5*5*6+1)*1     1516*10*10
    mpool2      10*10*16    2*2*16      5*5*16          -               -                   -
    conv3       5*5*16      5*5*120     1*120           1*120           (5*5*16+1)*120      (5*5*16+1)*120*1*1
    fc4         1*120       120*84      1*84            1*84            (120+1)*84          (120+1)*84
    fc5         1*84        84*10       1*10            1*10            (84+1)*10           (84+1)*10
    '''

    def __init__(self):
        print("python:", sys.version)
        print("tensorflow:", tf.__version__)
        print("GPU:", tf.test.is_gpu_available())
        tf.device("/gpu:0")
        self.mnist = input_data.read_data_sets(
            MNIST_DATASET_PATH, one_hot=True)
        self.width = 28
        self.height = 28
        self.channel = 1
        self.num_classes = 10
        self.max_step = 5000
        self.epoch = 0
        self.train_step = 0
        self.epoch_num = 100
        self.batch_size = 500
        self.learning_rate = 0.001
        self.early_stopping = 10
        self.max_acc = 0
        self.min_loss = float("inf")
        self.acc_list = []
        self.loss_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.model_path = LeNet_5_MODEL_PATH
        self.model_name = LeNet_5_MODEL_NAME
        self.model_record = LeNet_5_MODEL_PATH + LeNet_5_MODEL_RECORD
        self.train_record = LeNet_5_MODEL_PATH + LeNet_5_TRAIN_RECORD
        self.tensorboard = False
        if self.tensorboard:
            self.tensorboard_path = LeNet_5_TENSORBOARD_PATH
            self.feature_map = {}
            self.feature_map_limit = {}

        self.build()
        # self.build_with_layers()
        self.init()

    def build(self):
        with tf.name_scope("input"):
            self.input_images = tf.placeholder(
                dtype=tf.float32,
                shape=[
                    None,
                    self.height,
                    self.width,
                    self.channel],
                name="input_images")
            self.input_labels = tf.placeholder(
                dtype=tf.float32, shape=[
                    None, self.num_classes], name="input_labels")

        with tf.name_scope("conv1"):
            self.filter1 = tf.Variable(
                tf.truncated_normal([5, 5, 1, 6]), name="filter1")
            self.conv1 = tf.nn.conv2d(
                self.input_images, self.filter1, strides=[
                    1, 1, 1, 1], padding='SAME')
            self.bias1 = tf.Variable(tf.truncated_normal([6]), name="bias1")
            self.conv1_out = tf.nn.sigmoid(self.conv1 + self.bias1)
            if self.tensorboard:
                tf.summary.histogram("filter1", self.filter1)
                tf.summary.histogram("conv1", self.conv1)
                tf.summary.histogram("bias1", self.bias1)
                tf.summary.histogram("conv1_out", self.conv1_out)
                self.feature_map["conv1"] = self.conv1_out
                self.feature_map_limit["conv1"] = 6

        with tf.name_scope("mpool1"):
            self.mpool1_out = tf.nn.max_pool(
                self.conv1_out, ksize=[
                    1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
            if self.tensorboard:
                tf.summary.histogram("mpool1_out", self.mpool1_out)

        with tf.name_scope("conv2"):
            self.filter2 = tf.Variable(
                tf.truncated_normal([5, 5, 6, 16]), name="filter2")
            self.conv2 = tf.nn.conv2d(
                self.mpool1_out, self.filter2, strides=[
                    1, 1, 1, 1], padding='SAME')
            self.bias2 = tf.Variable(tf.truncated_normal([16]), name="bias2")
            self.conv2_out = tf.nn.sigmoid(self.conv2 + self.bias2)
            if self.tensorboard:
                tf.summary.histogram("filter2", self.filter2)
                tf.summary.histogram("conv2", self.conv2)
                tf.summary.histogram("bias2", self.bias2)
                tf.summary.histogram("conv2_out", self.conv2_out)
                self.feature_map["conv2"] = self.conv2_out
                self.feature_map_limit["conv2"] = 16

        with tf.name_scope("mpool2"):
            self.mpool2_out = tf.nn.max_pool(
                self.conv2_out, ksize=[
                    1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
            if self.tensorboard:
                tf.summary.histogram("mpool2_out", self.mpool2_out)

        with tf.name_scope("conv3"):
            self.filter3 = tf.Variable(
                tf.truncated_normal([5, 5, 16, 120]), name="filter3")
            self.conv3 = tf.nn.conv2d(
                self.mpool2_out, self.filter3, strides=[
                    1, 5, 5, 1], padding='VALID')
            self.bias3 = tf.Variable(tf.truncated_normal([120]), name="bias3")
            self.conv3_out = tf.nn.sigmoid(self.conv3 + self.bias3)
            if self.tensorboard:
                tf.summary.histogram("filter3", self.filter3)
                tf.summary.histogram("conv3", self.conv3)
                tf.summary.histogram("bias3", self.bias3)
                tf.summary.histogram("conv3_out", self.conv3_out)
                self.feature_map["conv3"] = self.conv3_out
                self.feature_map_limit["conv3"] = 120

        with tf.name_scope("fc4"):
            self.conv3_out_flat = tf.reshape(self.conv3_out, [-1, 120])
            self.fc4_w = tf.Variable(
                tf.truncated_normal([120, 84]), name="fc4_w")
            self.fc4_b = tf.Variable(tf.truncated_normal([84]), name="fc4_b")
            self.fc4_out = tf.nn.sigmoid(
                tf.matmul(
                    self.conv3_out_flat,
                    self.fc4_w) +
                self.fc4_b)
            if self.tensorboard:
                tf.summary.histogram("fc4_w", self.fc4_w)
                tf.summary.histogram("fc4_b", self.fc4_b)
                tf.summary.histogram("fc4_out", self.fc4_out)
                self.feature_map["fc4"] = self.fc4_out
                self.feature_map_limit["fc4"] = 84

        with tf.name_scope("fc5"):
            self.fc5_w = tf.Variable(tf.truncated_normal(
                [84, self.num_classes]), name="fc5_w")
            self.fc5_b = tf.Variable(tf.truncated_normal(
                [self.num_classes]), name="fc5_b")
            self.output = tf.nn.softmax(
                tf.matmul(
                    self.fc4_out,
                    self.fc5_w) +
                self.fc5_b)
            if self.tensorboard:
                tf.summary.histogram("fc5_w", self.fc5_w)
                tf.summary.histogram("fc5_b", self.fc5_b)
                tf.summary.histogram("fc5_out", self.output)
                self.feature_map["fc5"] = self.output
                self.feature_map_limit["fc5"] = self.num_classes

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.output, 1)
            self.ground_truth = tf.argmax(self.input_labels, 1)
            self.compare = tf.equal(self.prediction, self.ground_truth)
            self.accuracy = tf.reduce_mean(
                tf.cast(self.compare, dtype=tf.float32))
            if self.tensorboard:
                tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("loss"):
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_labels))
            self.loss = tf.reduce_mean((self.output - self.input_labels) ** 2)
            if self.tensorboard:
                tf.summary.scalar("loss", self.loss)

        with tf.name_scope("opt"):
            self.opt = tf.train.AdamOptimizer(
                self.learning_rate).minimize(
                self.loss)

    def build_with_layers(self):
        self.input_images = tf.placeholder(
            dtype=tf.float32,
            shape=[
                None,
                self.height,
                self.width,
                self.channel],
            name="input_images")
        self.input_labels = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.num_classes], name="input_labels")

        self.conv1 = tf.layers.Conv2D(
            filters=6, kernel_size=5, strides=(
                1, 1), kernel_initializer=tf.truncated_normal_initializer(
                stddev=tf.sqrt(
                    1 / 6)))
        self.apool1 = tf.layers.AveragePooling2D(
            pool_size=(2, 2), strides=(2, 2))
        self.conv2 = tf.layers.Conv2D(
            filters=16, kernel_size=5, strides=(
                1, 1), kernel_initializer=tf.truncated_normal_initializer(
                stddev=tf.sqrt(
                    1 / 16)))
        self.apool2 = tf.layers.AveragePooling2D(
            pool_size=(2, 2), strides=(2, 2))
        self.fc3 = tf.layers.Dense(
            120, kernel_initializer=tf.truncated_normal_initializer(
                stddev=tf.sqrt(
                    1 / 120)))
        self.fc4 = tf.layers.Dense(
            84, kernel_initializer=tf.truncated_normal_initializer(
                stddev=tf.sqrt(
                    1 / 84)))
        self.fc5 = tf.layers.Dense(
            self.num_classes,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=tf.sqrt(
                    1 / self.num_classes)))

        self.conv1_out = tf.nn.sigmoid(self.conv1(self.input_images))
        self.apool1_out = self.apool1(self.conv1_out)
        self.conv2_out = tf.nn.sigmoid(self.conv2(self.apool1_out))
        self.apool2_out = self.apool2(self.conv2_out)
        self.apool2_out_flat = tf.reshape(self.apool2_out, shape=[-1, 256])
        self.fc3_out = tf.nn.sigmoid(self.fc3(self.apool2_out_flat))
        self.fc4_out = tf.nn.sigmoid(self.fc4(self.fc3_out))
        self.output = tf.nn.softmax(self.fc5(self.fc4_out))

        self.prediction = tf.argmax(self.output, 1)
        self.ground_truth = tf.argmax(self.input_labels, 1)
        self.compare = tf.equal(self.prediction, self.ground_truth)
        self.accuracy = tf.reduce_mean(tf.cast(self.compare, dtype=tf.float32))
        self.loss = tf.reduce_mean((self.output - self.input_labels) ** 2)
        self.opt = tf.train.AdamOptimizer(
            self.learning_rate).minimize(
            self.loss)

    def init(self):
        self.sess = tf.Session()
        if self.tensorboard:
            if not tf.gfile.Exists(self.tensorboard_path):
                tf.gfile.MakeDirs(self.tensorboard_path)
            self.writer = tf.summary.FileWriter(
                self.tensorboard_path, self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=1)
        if not tf.gfile.Exists(self.model_path):
            tf.gfile.MakeDirs(self.model_path)
        latest_snap_shot = tf.train.latest_checkpoint(self.model_path)
        if latest_snap_shot:
            with open(self.model_record, 'r') as fr:
                model_json = json.load(fr)
                self.epoch = int(model_json["epoch"])
                self.train_step = int(model_json["train_step"])
                self.max_acc = float(model_json["accuracy"])
                self.min_loss = float(model_json["loss"])

            print(
                "Load model: epoch = {}，train_step = {}, acc = {:.6f}, loss = {:.6f}".format(
                    self.epoch,
                    self.train_step,
                    self.max_acc,
                    self.min_loss))
            self.saver.restore(self.sess, latest_snap_shot)
        else:
            print("Init with global_variables_initializer")
            self.sess.run(tf.global_variables_initializer())

    def save(self):
        print(
            "Save model: epoch = {}, train_step = {}, acc = {:.6f}, loss = {:.6f}".format(
                self.epoch,
                self.train_step,
                self.max_acc,
                self.min_loss))
        self.saver.save(
            self.sess,
            self.model_path +
            self.model_name,
            global_step=self.train_step)

        model_json = {
            "epoch": str(
                self.epoch), "train_step": str(
                self.train_step), "accuracy": str(
                self.max_acc), "loss": str(
                    self.min_loss)}
        with open(self.model_record, 'w') as fw:
            json.dump(model_json, fw)

    def draw(self):
        # plt.figure(figsize=(80, 60))
        plt.subplot(2, 2, 1)
        plt.title("train accuracy")
        plt.plot(self.acc_list)
        plt.subplot(2, 2, 2)
        plt.title("train loss")
        plt.plot(self.loss_list)
        plt.subplot(2, 2, 3)
        plt.title("validate accuracy")
        plt.plot(self.val_acc_list)
        plt.subplot(2, 2, 4)
        plt.title("validate loss")
        plt.plot(self.val_loss_list)
        plt.tight_layout()
        plt.savefig(self.train_record)
        # plt.show()

    def validate(self):
        val_images_flat, val_labels = self.mnist.test.next_batch(
            self.batch_size)
        val_images = val_images_flat.reshape(
            [-1, self.height, self.width, self.channel])
        val_acc, val_loss = self.sess.run(fetches=[self.accuracy, self.loss], feed_dict={
                                          self.input_images: val_images, self.input_labels: val_labels})
        return val_acc, val_loss

    def train(self):
        train_start = datetime.now()
        no_improvement = 0
        while self.train_step < self.max_step:
            epoch_start = datetime.now()
            self.train_step += 1
            self.learning_rate = 0.01 if self.train_step < 10 else (
                0.001 if self.train_step < 10000 else 0.0001)
            train_images_flat, train_labels = self.mnist.train.next_batch(
                self.batch_size)
            train_images = train_images_flat.reshape(
                [-1, self.height, self.width, self.channel])
            acc, loss, _ = self.sess.run(
                fetches=[
                    self.accuracy, self.loss, self.opt], feed_dict={
                    self.input_images: train_images, self.input_labels: train_labels})
            self.acc_list.append(acc)
            self.loss_list.append(loss)
            if self.train_step % self.epoch_num == 0:
                self.epoch += 1
                if self.tensorboard:
                    self.writer.add_summary(
                        self.sess.run(
                            tf.summary.merge_all(),
                            feed_dict={
                                self.input_images: train_images,
                                self.input_labels: train_labels}),
                        self.train_step)
                val_acc, val_loss = self.validate()
                self.val_acc_list.append(val_acc)
                self.val_loss_list.append(val_loss)
                print("Epoch[{:03d}]: val_acc = {:.6f}, val_loss = {:.6f}, taken = {}".format(
                    self.epoch, val_acc, val_loss, datetime.now() - epoch_start))
                if val_acc > self.max_acc and val_loss < self.min_loss:
                    self.max_acc = val_acc
                    self.min_loss = val_loss
                    no_improvement = 0
                    self.save()
                    if self.max_acc >= 99.999999 and self.min_loss <= 0.000001:
                        print(
                            "Train: stopped, would not improve, total {} epochs.".format(
                                self.epoch))
                        break
                else:
                    no_improvement += 1
                    if no_improvement < self.early_stopping:
                        print(
                            "Train: not improved in last {} epochs".format(no_improvement))
                    else:
                        print(
                            "Train: stopped, no more improvement since {} epochs".format(no_improvement))
                        break
        self.draw()
        if self.tensorboard:
            self.writer.close()
        print("Train: total_taken =", datetime.now() - train_start)

    def infer(self):
        test_images_flat, test_labels = self.mnist.test.next_batch(1)
        test_images = test_images_flat.reshape(
            [-1, self.height, self.width, self.channel])
        output = self.sess.run(fetches=[self.output], feed_dict={
                               self.input_images: test_images})
        print(
            "infer: {} --- label: {}".format(np.argmax(output[0]), np.argmax(test_labels)))
        if self.tensorboard:
            feature_map_dict = self.sess.run(fetches=[self.feature_map], feed_dict={
                                             self.input_images: test_images})
            for k in feature_map_dict[0]:
                if len(feature_map_dict[0][k].shape) == 4:
                    for i in range(feature_map_dict[0][k].shape[0]):
                        feature_map = tf.transpose(tf.reshape(feature_map_dict[0][k][i: i + 1, :, :, :self.feature_map_limit[k]],
                                                              (1, feature_map_dict[0][k].shape[1], -1, 1)), [0, 2, 1, 3])
                        tf.summary.image("%s/%d" % (k, i), feature_map)
                elif len(feature_map_dict[0][k].shape) == 2:
                    for i in range(feature_map_dict[0][k].shape[0]):
                        feature_map = tf.transpose(tf.reshape(feature_map_dict[0][k][i: i + 1, :self.feature_map_limit[k]],
                                                              (1, feature_map_dict[0][k].shape[1], -1, 1)), [0, 2, 1, 3])
                        tf.summary.image("%s/%d" % (k, i), feature_map)
            self.writer.add_summary(
                self.sess.run(
                    tf.summary.merge_all(),
                    feed_dict={
                        self.input_images: test_images,
                        self.input_labels: test_labels}),
                0)
            self.writer.close()


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = LeNet_5()
    net.train()
    # net.infer()


if __name__ == "__main__":
    main()
