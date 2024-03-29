#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        probs,feature = net_g(data)
        test_loss += F.cross_entropy(probs, target,reduction='sum').item()
        y_pred = probs.data.max(1, keepdim=True)[1]

        correct += y_pred.eq(target.data.view_as(y_pred)).cpu().sum().item()

    test_loss = test_loss/len(data_loader.dataset)
    test_accuracy = (100.00 * correct / len(data_loader.dataset))

    return test_accuracy, test_loss