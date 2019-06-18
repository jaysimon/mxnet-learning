#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 ShenZhen Hian Speech S&T Co.,Ltd. All rights reserved.
# FileName : demo.py
# Author : Hou Wei
# Version : V1.0
# Date: 2019-02-21
# Description:
# History:

# import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time
import cv2


def create_data():
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    return mnist_train, mnist_test


def main():
    mnist_train, mnist_test = create_data()
    feature, label = mnist_train[0]
    print(feature.shape)
    print(label)
    # img = feature.asnumpy()
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    iBatchSize = 1024
    tranformer = gdata.vision.transforms.ToTensor()

    iNumWorkers = 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(tranformer),
                                  iBatchSize, shuffle=True)
    start = time.time()
    for X, y in train_iter:
        continue
    print("%.2f sec" % (time.time() - start))


if __name__ == "__main__":
    main()
