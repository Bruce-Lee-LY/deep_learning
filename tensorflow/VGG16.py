# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: VGG16

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
from tensorflow import contrib
import numpy as np
import matplotlib.pyplot as plt

from Caltech_256 import Caltech_256

VGG16_MODEL_PATH = "model/VGG16/"
VGG16_MODEL_NAME = "VGG16_train_step"
VGG16_MODEL_RECORD = "VGG16_model_record.json"
VGG16_TRAIN_RECORD = "VGG16_train_record.png"
VGG16_TENSORBOARD_PATH = "tensorboard/VGG16/"


class VGG16:
    '''
    layer       input           kernel          feature_map      neuron_num     param_num             connect_num
    conv1_1     224*224*3       3*3*3*64        224*224*64       224*224*64     (3*3*3+1)*64          (3*3*3+1)*64*224*224
    conv1_2     224*224*64      3*3*64*64       224*224*64       224*224*64     (3*3*64+1)*64         (3*3*64+1)*64*224*224
    mpool1      224*224*64      1*2*2*1         112*112*64       -              -                     -
    conv2_1     112*112*64      3*3*64*128      112*112*128      112*112*128    (3*3*64+1)*128        (3*3*64+1)*128*112*112
    conv2_2     112*112*128     3*3*128*128     112*112*120      112*112*120    (3*3*128+1)*128       (3*3*128+1)*128*112*112
    mpool2      112*112*128     1*2*2*1         56*56*128        -              -                     -
    conv3_1     56*56*128       3*3*128*256     56*56*256        56*56*256      (3*3*128+1)*256       (3*3*128+1)*256*56*56
    conv3_2     56*56*256       3*3*256*256     56*56*256        56*56*256      (3*3*256+1)*256       (3*3*256+1)*256*56*56
    conv3_3     56*56*256       3*3*256*256     56*56*256        56*56*256      (3*3*256+1)*256       (3*3*256+1)*256*56*56
    mpool3      56*56*256       1*2*2*1         28*28*256        -              -                     -
    conv4_1     28*28*256       3*3*256*512     28*28*512        28*28*512      (3*3*256+1)*512       (3*3*256+1)*512*28*28
    conv4_2     28*28*512       3*3*512*512     28*28*512        28*28*512      (3*3*512+1)*512       (3*3*512+1)*512*28*28
    conv4_3     28*28*512       3*3*512*512     28*28*512        28*28*512      (3*3*512+1)*512       (3*3*512+1)*512*28*28
    mpool4      28*28*512       1*2*2*1         14*14*512        -              -                     -
    conv5_1     14*14*512       3*3*512*512     14*14*512        14*14*512      (3*3*512+1)*512       (3*3*512+1)*512*14*14
    conv5_2     14*14*512       3*3*512*512     14*14*512        14*14*512      (3*3*512+1)*512       (3*3*512+1)*512*14*14
    conv5_3     14*14*512       3*3*512*512     14*14*512        14*14*512      (3*3*512+1)*512       (3*3*512+1)*512*14*14
    mpool5      14*14*512       1*2*2*1         7*7*512          -              -                     -
    fc6         7*7*512         (7*7*512)*4096  1*4096           4096           (7*7*512+1)*4096      (7*7*512+1)*4096
    drop6       1*4096          -               1*4096           -              -                     -
    fc7         1*4096          4096*4096       1*4096           4096           (4096+1)*4096         (4096+1)*4096
    drop7       1*4096          -               1*4096           -              -                     -
    fc8         1*4096          4096*4096       1*num_classes    num_classes    (4096+1)*num_classes  (4096+1)*num_classes
    '''

    def __init__(self):
        print("Python:", sys.version)
        print("Tensorflow:", tf.__version__)
        print("GPU:", tf.test.is_gpu_available())
        tf.device("/gpu:0")
        self.width = 224
        self.height = 224
        self.channel = 3
        self.train_batch_size = 256
        self.val_batch_size = 256
        self.test_batch_size = 1
        self.caltch_256 = Caltech_256(
            self.width,
            self.height,
            self.channel,
            self.train_batch_size,
            self.val_batch_size,
            self.test_batch_size)
        self.num_classes = self.caltch_256.caltech_classes
        self.dropout_rate = 0.1
        self.epoch = 0
        self.train_step = 0
        self.max_epoch = 1000
        self.learning_rate = 0.0001
        self.early_stopping = 32
        self.max_acc = 0
        self.min_loss = float("inf")
        self.acc_list = []
        self.loss_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.model_path = VGG16_MODEL_PATH
        self.model_name = VGG16_MODEL_NAME
        self.model_record = VGG16_MODEL_PATH + VGG16_MODEL_RECORD
        self.train_record = VGG16_MODEL_PATH + VGG16_TRAIN_RECORD
        self.tensorboard = False
        if self.tensorboard:
            self.tensorboard_path = VGG16_TENSORBOARD_PATH
            self.feature_map = {}
            self.feature_map_limit = {}

        # self.build()
        self.build_simple()
        self.init()

    def _conv(self, seq, shape, strides, input):
        with tf.name_scope("conv" + seq):
            filter = tf.get_variable(
                "filter" + seq,
                shape=shape,
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input, filter, strides=strides, padding='SAME')
            # bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[shape[-1]]), name="bias" + seq)
            bias = tf.get_variable("bias" + seq,
                                   shape=[shape[-1]],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv_out = tf.nn.relu(tf.nn.bias_add(conv, bias))
            if self.tensorboard:
                tf.summary.histogram("filter" + seq, filter)
                tf.summary.histogram("conv" + seq, conv)
                tf.summary.histogram("bias" + seq, bias)
                tf.summary.histogram("conv_out" + seq, conv_out)
                self.feature_map["conv" + seq] = conv_out
                self.feature_map_limit["conv" + seq] = shape[-1]
            return conv_out

    def _mpool(self, seq, ksize, strides, input):
        with tf.name_scope("mpool" + seq):
            mpool_out = tf.nn.max_pool(
                input,
                ksize=ksize,
                strides=strides,
                padding='SAME',
                name="mpool" + seq)
            if self.tensorboard:
                tf.summary.histogram("mpool_out" + seq, mpool_out)
            return mpool_out

    def _fc(self, seq, shape, input):
        with tf.name_scope("fc" + seq):
            fc_w = tf.get_variable(
                "fc_w" + seq,
                shape=shape,
                dtype=tf.float32,
                initializer=contrib.layers.xavier_initializer())
            # fc_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[shape[-1]]), name="fc_b" + seq)
            fc_b = tf.get_variable("fc_b" + seq,
                                   shape=[shape[-1]],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            fc_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, fc_w), fc_b))
            if self.tensorboard:
                tf.summary.histogram("fc_w" + seq, fc_w)
                tf.summary.histogram("fc_b" + seq, fc_b)
                tf.summary.histogram("fc_out" + seq, fc_out)
                self.feature_map["fc" + seq] = fc_out
                self.feature_map_limit["fc" + seq] = shape[-1]
            return fc_out

    def _dropout(self, seq, rate, input):
        with tf.name_scope("drop" + seq):
            drop_out = tf.nn.dropout(input, rate=rate)
            if self.tensorboard:
                tf.summary.histogram("drop_out" + seq, drop_out)
            return drop_out

    def _softmax(self, seq, input):
        with tf.name_scope('softmax' + seq):
            softmax_out = tf.nn.softmax(input)
            if self.tensorboard:
                tf.summary.histogram("softmax_out" + seq, softmax_out)
            return softmax_out

    def build_simple(self):
        with tf.name_scope("input"):
            self.input_images = tf.placeholder(
                tf.float32,
                shape=[
                    None,
                    self.height,
                    self.width,
                    self.channel],
                name='input_images')
            self.input_labels = tf.placeholder(
                tf.float32, shape=[
                    None, self.num_classes], name='input_labels')

        self.conv_out1 = self._conv(
            "1", [
                3, 3, 3, 64], [
                1, 1, 1, 1], self.input_images)
        self.mpool_out1 = self._mpool(
            "1", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out1)

        self.conv_out2 = self._conv(
            "2", [
                3, 3, 64, 128], [
                1, 1, 1, 1], self.mpool_out1)
        self.mpool_out2 = self._mpool(
            "2", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out2)

        self.conv_out3 = self._conv(
            "3", [
                3, 3, 128, 256], [
                1, 1, 1, 1], self.mpool_out2)
        self.mpool_out3 = self._mpool(
            "3", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out3)

        self.conv_out4 = self._conv(
            "4", [
                3, 3, 256, 512], [
                1, 1, 1, 1], self.mpool_out3)
        self.mpool_out4 = self._mpool(
            "4", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out4)

        self.conv_out5 = self._conv(
            "5", [
                3, 3, 512, 512], [
                1, 1, 1, 1], self.mpool_out4)
        self.mpool_out5 = self._mpool(
            "5", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out5)

        self.fc_out6 = self._fc(
            "6", [7 * 7 * 512, self.num_classes], tf.reshape(self.mpool_out5, [-1, 7 * 7 * 512]))
        self.output = self._softmax("6", self.fc_out6)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.output, 1)
            self.ground_truth = tf.argmax(self.input_labels, 1)
            self.compare = tf.equal(self.prediction, self.ground_truth)
            self.accuracy = tf.reduce_mean(
                tf.cast(self.compare, dtype=tf.float32))
            if self.tensorboard:
                tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.input_labels))
            # self.loss = tf.reduce_mean((self.output - self.input_labels) ** 2)
            if self.tensorboard:
                tf.summary.scalar("loss", self.loss)

        with tf.name_scope("opt"):
            self.opt = tf.train.AdamOptimizer(
                self.learning_rate).minimize(
                self.loss)
            # self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def build(self):
        with tf.name_scope("input"):
            self.input_images = tf.placeholder(
                tf.float32,
                shape=[
                    None,
                    self.height,
                    self.width,
                    self.channel],
                name='input_images')
            self.input_labels = tf.placeholder(
                tf.float32, shape=[
                    None, self.num_classes], name='input_labels')

        self.conv_out1_1 = self._conv(
            "1_1", [
                3, 3, 3, 64], [
                1, 1, 1, 1], self.input_images)
        self.conv_out1_2 = self._conv(
            "1_2", [
                3, 3, 64, 64], [
                1, 1, 1, 1], self.conv_out1_1)
        self.mpool_out1 = self._mpool(
            "1", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out1_2)

        self.conv_out2_1 = self._conv(
            "2_1", [
                3, 3, 64, 128], [
                1, 1, 1, 1], self.mpool_out1)
        self.conv_out2_2 = self._conv(
            "2_2", [
                3, 3, 128, 128], [
                1, 1, 1, 1], self.conv_out2_1)
        self.mpool_out2 = self._mpool(
            "2", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out2_2)

        self.conv_out3_1 = self._conv(
            "3_1", [
                3, 3, 128, 256], [
                1, 1, 1, 1], self.mpool_out2)
        self.conv_out3_2 = self._conv(
            "3_2", [
                3, 3, 256, 256], [
                1, 1, 1, 1], self.conv_out3_1)
        self.conv_out3_3 = self._conv(
            "3_3", [
                3, 3, 256, 256], [
                1, 1, 1, 1], self.conv_out3_2)
        self.mpool_out3 = self._mpool(
            "3", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out3_3)

        self.conv_out4_1 = self._conv(
            "4_1", [
                3, 3, 256, 512], [
                1, 1, 1, 1], self.mpool_out3)
        self.conv_out4_2 = self._conv(
            "4_2", [
                3, 3, 512, 512], [
                1, 1, 1, 1], self.conv_out4_1)
        self.conv_out4_3 = self._conv(
            "4_3", [
                3, 3, 512, 512], [
                1, 1, 1, 1], self.conv_out4_2)
        self.mpool_out4 = self._mpool(
            "4", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out4_3)

        self.conv_out5_1 = self._conv(
            "5_1", [
                3, 3, 512, 512], [
                1, 1, 1, 1], self.mpool_out4)
        self.conv_out5_2 = self._conv(
            "5_2", [
                3, 3, 512, 512], [
                1, 1, 1, 1], self.conv_out5_1)
        self.conv_out5_3 = self._conv(
            "5_3", [
                3, 3, 512, 512], [
                1, 1, 1, 1], self.conv_out5_2)
        self.mpool_out5 = self._mpool(
            "5", [
                1, 2, 2, 1], [
                1, 2, 2, 1], self.conv_out5_3)

        self.fc_out6 = self._fc(
            "6", [7 * 7 * 512, 4096], tf.reshape(self.mpool_out5, [-1, 7 * 7 * 512]))
        self.drop_out6 = self._dropout("6", self.dropout_rate, self.fc_out6)

        self.fc_out7 = self._fc("7", [4096, 4096], self.drop_out6)
        self.drop_out7 = self._dropout("7", self.dropout_rate, self.fc_out7)

        self.fc_out8 = self._fc("8", [4096, self.num_classes], self.drop_out7)
        self.output = self._softmax("8", self.fc_out8)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.output, 1)
            self.ground_truth = tf.argmax(self.input_labels, 1)
            self.compare = tf.equal(self.prediction, self.ground_truth)
            self.accuracy = tf.reduce_mean(
                tf.cast(self.compare, dtype=tf.float32))
            if self.tensorboard:
                tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.input_labels))
            # self.loss = tf.reduce_mean((self.output - self.input_labels) ** 2)
            if self.tensorboard:
                tf.summary.scalar("loss", self.loss)

        with tf.name_scope("opt"):
            self.opt = tf.train.AdamOptimizer(
                self.learning_rate).minimize(
                self.loss)
            # self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

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
        val_start = datetime.now()
        self.caltch_256.reset_val_batch()
        batch = 0
        acc_sum = 0.0
        loss_sum = 0.0
        while self.caltch_256.can_get_val_batch():
            batch_start = datetime.now()
            batch += 1
            self.val_batch = self.caltch_256.get_val_batch()
            acc, loss = self.sess.run(fetches=[self.accuracy, self.loss], feed_dict={
                                      self.input_images: self.val_batch.images, self.input_labels: self.val_batch.labels})
            acc_sum += acc
            loss_sum += loss
            print("Batch[{:03d}]: val_acc = {:.6f}, val_loss = {:.6f}, taken = {}".format(
                batch, acc, loss, datetime.now() - batch_start))
        val_acc = acc_sum / batch
        val_loss = loss_sum / batch
        print("Validate: val_acc = {:.6f}, val_loss = {:.6f}, total_taken = {}".format(
            val_acc, val_loss, datetime.now() - val_start))
        return val_acc, val_loss

    def train(self):
        train_start = datetime.now()
        no_improvement = 0
        while self.epoch < self.max_epoch:
            epoch_start = datetime.now()
            self.epoch += 1
            self.caltch_256.reset_train_batch()
            batch = 0
            while self.caltch_256.can_get_train_batch():
                batch_start = datetime.now()
                batch += 1
                self.learning_rate = 0.0001 if self.train_step < 1000000 else (
                    0.00001 if self.train_step < 10000000 else 0.000001)
                self.train_step += 1
                self.train_batch = self.caltch_256.get_train_batch()
                acc, loss, _ = self.sess.run(fetches=[self.accuracy, self.loss, self.opt], feed_dict={
                                             self.input_images: self.train_batch.images, self.input_labels: self.train_batch.labels})
                self.acc_list.append(acc)
                self.loss_list.append(loss)
                print(
                    "Epoch[{:03d}]-Batch[{:03d}]: acc = {:.6f}, loss = {:.6f}, taken = {}".format(
                        self.epoch,
                        batch,
                        acc,
                        loss,
                        datetime.now() -
                        batch_start))

            if self.tensorboard:
                self.writer.add_summary(
                    self.sess.run(
                        tf.summary.merge_all(),
                        feed_dict={
                            self.input_images: self.train_batch.images,
                            self.input_labels: self.train_batch.labels}),
                    self.train_step)
            val_acc, val_loss = self.validate()
            self.val_acc_list.append(val_acc)
            self.val_loss_list.append(val_loss)
            print("Epoch[{:03d}]: total_taken = {}".format(
                self.epoch, datetime.now() - epoch_start))
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
        test_batch = self.caltch_256.get_test_batch()
        output = self.sess.run(fetches=[self.output], feed_dict={
                               self.input_images: test_batch.images})
        print(
            "infer: {} --- label: {}".format(
                np.argmax(
                    output[0]), np.argmax(
                    test_batch.labels)))
        if self.tensorboard:
            feature_map_dict = self.sess.run(fetches=[self.feature_map], feed_dict={
                                             self.input_images: test_batch.images})
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
                        self.input_images: test_batch.images,
                        self.input_labels: test_batch.labels}),
                0)
            self.writer.close()


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = VGG16()
    net.train()
    # net.infer()


if __name__ == "__main__":
    main()
