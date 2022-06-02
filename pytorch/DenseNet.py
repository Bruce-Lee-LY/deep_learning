# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: DenseNet

#!/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import sys
import os
from datetime import datetime
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.checkpoint as cp
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

CIFAR10_DATASET_PATH = "dataset/"
DenseNet_MODEL_PATH = "model/DenseNet/"
DenseNet_MODEL_NAME = "DenseNet_train_step-"
DenseNet_MODEL_RECORD = "DenseNet_model_record.json"
DenseNet_TRAIN_RECORD = "DenseNet_train_record.png"


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_features,
            growth_rate,
            bn_size,
            drop_rate,
            efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size *
                growth_rate,
                kernel_size=1,
                stride=1,
                bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                bn_size *
                growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(
                prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features,
                p=self.drop_rate,
                training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(
            self,
            num_layers,
            num_input_features,
            bn_size,
            growth_rate,
            drop_rate,
            efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features +
                i *
                growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet_net(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(
            self,
            growth_rate=12,
            block_config=(
                16,
                16,
                16),
            compression=0.5,
            num_init_features=24,
            bn_size=4,
            drop_rate=0,
            num_classes=10,
            small_inputs=True,
            efficient=False):

        super(DenseNet_net, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(
                3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)), ]))
        else:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(
                3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ]))
            self.features.add_module(
                'norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module(
                'pool0',
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers,
                num_features,
                bn_size,
                growth_rate,
                drop_rate,
                efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_features, int(
                        num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class DenseNet():
    def __init__(self, seed=0):
        print("Python:", sys.version)
        print("PyTorch:", torch.__version__)
        print("TorchVision:", torchvision.__version__)
        print("GPU:", torch.cuda.is_available())
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(seed)
        self.epoch = 0
        self.train_step = 0
        self.max_epoch = 300
        self.early_stopping = 6
        self.max_acc = 0
        self.min_loss = float("inf")
        self.acc_list = []
        self.loss_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.dataset_path = CIFAR10_DATASET_PATH
        self.model_path = DenseNet_MODEL_PATH
        self.model_name = DenseNet_MODEL_NAME
        self.model_record = DenseNet_MODEL_PATH + DenseNet_MODEL_RECORD
        self.train_record = DenseNet_MODEL_PATH + DenseNet_TRAIN_RECORD
        self.width = 32
        self.height = 32
        self.channel = 3
        self.num_classes = 10
        self.train_batch_size = 256
        self.val_batch_size = 256
        self.test_batch_size = 1
        self.val_fraction = 0.1
        self.padding = 4
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        self.num_workers = 5
        self.pin_memory = True
        self.drop_last = True
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.nesterov = True
        self.milestones = [int(0.5 * self.max_epoch),
                           int(0.75 * self.max_epoch)]
        self.gamma = 0.1

        self.produce()
        self.build()
        self.init()

    def produce(self):
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(
                    (self.height,
                     self.width),
                    self.padding),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.mean,
                    self.std)])
        self.val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        self.train_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=True,
            transform=self.train_transform,
            download=True)
        self.val_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=True,
            transform=self.val_transform,
            download=False)
        self.test_dataset = datasets.CIFAR10(
            self.dataset_path,
            train=False,
            transform=self.test_transform,
            download=False)
        self.train_num = int(len(self.train_dataset) * (1 - self.val_fraction))
        self.val_num = int(len(self.train_dataset) * self.val_fraction)
        self.test_num = int(len(self.test_dataset))
        print("CIFAR10: train_num = {}, train_batch_size = {}, train_batch = {}".format(
            self.train_num, self.train_batch_size, self.train_num // self.train_batch_size))
        print("CIFAR10: val_num = {}, val_batch_size = {}, val_batch = {}".format(
            self.val_num, self.val_batch_size, self.val_num // self.val_batch_size))
        print("CIFAR10: test_num = {}, test_batch_size = {}, test_batch = {}".format(
            self.test_num, self.test_batch_size, self.test_num // self.test_batch_size))
        self.indices = torch.randperm(len(self.train_dataset))
        self.train_indices = self.indices[:len(self.indices) - self.val_num]
        self.val_indices = self.indices[len(self.indices) - self.val_num:]
        self.train_dataset = torch.utils.data.Subset(
            self.train_dataset, self.train_indices)
        self.val_dataset = torch.utils.data.Subset(
            self.val_dataset, self.val_indices)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last)

    def build(self):
        self.model = DenseNet_net(num_classes=self.num_classes).to(self.device)
        print(self.model)
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), self.learning_rate, self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            self.learning_rate)
        print(self.optimizer)
        self.lr_schedule = MultiStepLR(
            self.optimizer, self.milestones, gamma=self.gamma)

    def init(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        files = os.listdir(self.model_path)
        if files:
            for flie in files:
                if os.path.splitext(flie)[-1] == ".pth":
                    latest_pth = torch.load(self.model_path + flie)
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
                    self.model.load_state_dict(latest_pth['model'])
                    self.optimizer.load_state_dict(latest_pth['optimizer'])
                    self.lr_schedule.load_state_dict(latest_pth['lr_schedule'])
                    break

    def save(self):
        files = os.listdir(self.model_path)
        if files:
            for flie in files:
                if os.path.splitext(flie)[-1] == ".pth":
                    os.remove(self.model_path + flie)
        print(
            "Save model: epoch = {}, train_step = {}, acc = {:.6f}, loss = {:.6f}".format(
                self.epoch,
                self.train_step,
                self.max_acc,
                self.min_loss))
        self.checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_schedule": self.lr_schedule.state_dict()}
        torch.save(self.checkpoint, self.model_path +
                   self.model_name + str(self.train_step) + ".pth")

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
        self.model.eval()
        batch = 0
        acc_sum = 0.0
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                batch_start = datetime.now()
                batch += 1
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(
                    labels.view_as(pred)).sum().item() / self.val_batch_size
                loss = self.criterion(output, labels).item()
                acc_sum += acc
                loss_sum += loss
                print(
                    "Batch[{:03d}]: val_acc = {:.6f}, val_loss = {:.6f}, taken = {}".format(
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
            self.model.train()
            batch = 0
            for images, labels in self.train_loader:
                batch_start = datetime.now()
                batch += 1
                self.train_step += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(labels.view_as(pred)).sum(
                ).item() / self.train_batch_size
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.acc_list.append(acc)
                self.loss_list.append(loss.item())
                print(
                    "Epoch[{:03d}]-Batch[{:03d}]: acc = {:.6f}, loss = {:.6f}, taken = {}".format(
                        self.epoch,
                        batch,
                        acc,
                        loss.item(),
                        datetime.now() -
                        batch_start))
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
            self.lr_schedule.step()
        self.draw()
        print("Train: total_taken =", datetime.now() - train_start)

    def infer(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                pred = output.argmax(dim=1, keepdim=True)
                print("infer: {} --- label: {}".format(pred.item(), labels.item()))
                break


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = DenseNet()
    net.train()
    # net.infer()


if __name__ == "__main__":
    main()
