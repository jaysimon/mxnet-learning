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
import random


def create_data(iNum=1000, fW1=2, fW2=-3.4, fB=4.2):
    ndW = nd.array([fW1, fW2])
    ndX = nd.random.normal(scale=1, shape=[iNum, 2])
    ndY = nd.dot(ndX, nd.transpose(ndW)) + fB
    ndY += nd.random.normal(scale=0.01, shape=ndY.shape) 
    return ndX, ndY


def get_batch(ndX, ndY, iBatchSize):
    iNum = ndX.shape[0]
    lIndice = list(range(iNum))
    random.shuffle(lIndice)
    # print(lIndice)
    for iIndex in range(0, len(lIndice), iBatchSize):
        ndJ = nd.array(lIndice[iIndex:min(iIndex + iBatchSize, iNum)])
        # print(ndX.take(ndJ))
        # print(ndY.take(ndJ))
        yield ndX.take(ndJ), ndY.take(ndJ)


def forward(ndX, ndW, ndB):
    return nd.dot(ndX, nd.transpose(ndW)) + ndB


def loss(ndPred, ndTrue):
    return ((ndPred - ndTrue) ** 2) / 2


def backward(params, fLR=0.1, iBatchSize=10):
    for param in params:
        param[:] = param - param.grad * fLR / iBatchSize


def train(ndX, ndY):
    fLR = 0.03
    iEpoch = 3
    iBatchSize = 10

    ndW = nd.random.normal(0.01, shape=[2, ])
    ndB = nd.zeros(shape=(1,))
    ndW.attach_grad()
    ndB.attach_grad()
    for iIndex in range(iEpoch):
        for ndBatchX, ndBatchY in get_batch(ndX, ndY, iBatchSize):
            with autograd.record():
                ndPredY = forward(ndBatchX, ndW, ndB)
                fLoss = loss(ndPredY, ndBatchY)
            fLoss.backward()
            backward([ndW, ndB], fLR=fLR, iBatchSize=iBatchSize)
        # print("params:{}".format([ndW, ndB]))
        train_l = loss(forward(ndX, ndW, ndB), ndY)
        # print(train_l.shape)
        print(
            "[Log] Epoch:%d Loss is %f\n" %
            (iIndex+1, (train_l.mean().asnumpy())))
    print(ndW)
    print(ndB)


def main():
    ndX, ndY = create_data(iNum=1000, fW1=2, fW2=-3.4, fB=4.2)
    ndA = nd.array([1.5, 2.4])
    ndB = nd.array([1, 5.1])
    print(loss(ndA, ndB))
    train(ndX, ndY)


if __name__ == "__main__":
    main()
