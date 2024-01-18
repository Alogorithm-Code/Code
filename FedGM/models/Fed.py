#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn, autograd
import numpy as np

#进行aggreagate
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
    	if 'num_batches_tracked' in k:
    		w_avg[k] = w[0][k]
    	else:
        	for i in range(1, len(w)):
        		w_avg[k] += w[i][k]
        	w_avg[k] = torch.div(w_avg[k],len(w))
    return w_avg

