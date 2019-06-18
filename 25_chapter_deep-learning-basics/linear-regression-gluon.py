#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 ShenZhen Hian Speech S&T Co.,Ltd. All rights reserved.
# FileName : demo.py
# Author : Hou Wei
# Version : V1.0
# Date: 2019-02-21
# Description:
# History:


from mxnet import autograd, nd, init
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet.gluon import Trainer



def create_data(iNumExample=1000, iNumInput=2,
                ndW=nd.array([2, -3.4]), ndB=4.2):
    ndX = nd.random.normal(scale=1, shape=(iNumExample, iNumInput))
    ndY = nd.dot(ndX, ndW) + ndB
    return ndX, ndY


def create_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    return net


def main():
    iBatchSize = 10
    iEpoch = 10

    ndX, ndY = create_data()
    # type:ArrayDataset
    adDataset = gdata.ArrayDataset(ndX, ndY)
    # type:DataLoader
    dlDataIter = gdata.DataLoader(adDataset, batch_size=iBatchSize,
                                  shuffle=True)
    net = create_net()
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    trainer = Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})

    for iIndex in range(iEpoch):
        for ndBatchX, ndBatchY in dlDataIter:
            with autograd.record():
                l = loss(net(ndBatchX), ndBatchY)
            l.backward()
            trainer.step(iBatchSize)
        l = loss(net(ndX), ndY)
        print("[Log] Epoch:%d Loss:%f" % (iIndex, l.mean().asnumpy()))
    print(net[0].weight.data())
    print(net[0].bias.data())


if __name__ == "__main__":
    main()
