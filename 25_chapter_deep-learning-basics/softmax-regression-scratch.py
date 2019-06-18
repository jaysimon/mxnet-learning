#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 ShenZhen Hian Speech S&T Co.,Ltd. All rights reserved.
# FileName : demo.py
# Author : Hou Wei
# Version : V1.0
# Date: 2019-02-21
# Description:
# History:


from mxnet import autograd, nd
from mxnet.gluon import data as gdata



def create_data(iBatchSize=256):
    tranformer = gdata.vision.transforms.ToTensor()
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    train_iter = gdata.DataLoader(mnist_train.transform_first(tranformer),
                                  iBatchSize, shuffle=True)
    test_iter = gdata.DataLoader(mnist_train.transform_first(tranformer),
                                  iBatchSize, shuffle=True)
    return train_iter, test_iter


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def softmax(ndX):
    ndX_exp = ndX.exp()
    partition = ndX_exp.sum(axis=1, keepdims=True)
    return ndX_exp/partition

def net(ndX, num_inputs, W, b):
    return softmax(nd.dot(ndX.reshape((-1, num_inputs)), W) + b)


def main():
    iBatchSize = 256

    train_iter, test_iter = create_data()
    num_inputs = 784
    num_outputs = 10
    W = nd.random.normal(scale=0.1, shape=(num_inputs, num_outputs))
    b = nd.zeros(num_outputs)
    W.attach_grad()
    b.attach_grad()

    ndX = nd.array([1, 2, 3, 4]).transpose()
    print(softmax(ndX))




if __name__ == "__main__":
    main()