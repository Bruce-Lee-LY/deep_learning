# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 22:07:31 on Tue, May 24, 2022
#
# Description: ResNet152

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.checkpoint as cp
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet152
import matplotlib.pyplot as plt

Caltech256_DATASET_PATH = "dataset/"
ResNet152_MODEL_PATH = "model/ResNet152/"
ResNet152_MODEL_NAME = "ResNet152_train_step-"
ResNet152_MODEL_RECORD = "ResNet152_model_record.json"
ResNet152_TRAIN_RECORD = "ResNet152_train_record.png"


class ResNet152():
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
        self.max_epoch = 500
        self.early_stopping = 10
        self.max_acc = 0
        self.min_loss = float("inf")
        self.acc_list = []
        self.loss_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.dataset_path = Caltech256_DATASET_PATH
        self.model_path = ResNet152_MODEL_PATH
        self.model_name = ResNet152_MODEL_NAME
        self.model_record = ResNet152_MODEL_PATH + ResNet152_MODEL_RECORD
        self.train_record = ResNet152_MODEL_PATH + ResNet152_TRAIN_RECORD
        self.width = 224
        self.height = 224
        self.channel = 3
        self.num_classes = 257
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.test_batch_size = 1
        self.val_fraction = 0.1
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.num_workers = 5
        self.pin_memory = True
        self.drop_last = True
        self.pretrained = False
        self.progress = True
        self.learning_rate = 0.01
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
                transforms.Resize(
                    (self.height,
                     self.width)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.repeat(
                        3,
                        1,
                        1) if x.shape[0] == 1 else x),
                transforms.Normalize(
                    self.mean,
                    self.std)])
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width)), transforms.ToTensor(), transforms.Lambda(
                    lambda x: x.repeat(
                        3, 1, 1) if x.shape[0] == 1 else x), transforms.Normalize(
                        self.mean, self.std)])
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width)), transforms.ToTensor(), transforms.Lambda(
                    lambda x: x.repeat(
                        3, 1, 1) if x.shape[0] == 1 else x), transforms.Normalize(
                        self.mean, self.std)])
        self.train_dataset = datasets.Caltech256(
            self.dataset_path, transform=self.train_transform, download=True)
        self.val_dataset = datasets.Caltech256(
            self.dataset_path, transform=self.val_transform, download=False)
        self.test_dataset = datasets.Caltech256(
            self.dataset_path, transform=self.test_transform, download=False)
        self.train_num = int(len(self.train_dataset) * (1 - self.val_fraction))
        self.val_num = int(len(self.train_dataset) * self.val_fraction)
        self.test_num = int(len(self.test_dataset))
        print("Caltech256: train_num = {}, train_batch_size = {}, train_batch = {}".format(
            self.train_num, self.train_batch_size, self.train_num // self.train_batch_size))
        print("Caltech256: val_num = {}, val_batch_size = {}, val_batch = {}".format(
            self.val_num, self.val_batch_size, self.val_num // self.val_batch_size))
        print("Caltech256: test_num = {}, test_batch_size = {}, test_batch = {}".format(
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
        self.model = resnet152(
            self.pretrained,
            self.progress,
            num_classes=self.num_classes).to(
            self.device)
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
    net = ResNet152()
    net.train()
    # net.infer()


if __name__ == "__main__":
    main()
