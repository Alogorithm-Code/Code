#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import Counter

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    count_lists = []
    labels = np.array(dataset.targets)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        #compute each label numbers
        count_list=[0 for j in range(10)]
        #all_label = set([k for k in range(10)])
        #label = set(labels[list(dict_users[i])])
        #zero_label = list(all_label - label)
        #if len(zero_label) > 0:
        #    for element in zero_label:
        #        count_list[element] = 0
        label_number = Counter(labels[list(dict_users[i])])
        for key,value in label_number.items():
            count_list[key] = value
        count_lists.append(count_list)

    return dict_users,count_lists


def mnist_noniid_dirichlet(dataset, num_users, beta):
    #beta is smaller, daat is more un-balanced

    labels = np.array(dataset.targets)
    N = len(labels)
    dict_users = {}
    min_size = 0
    min_require_size = 10
    K = 10

    count_lists = []
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            #print(proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        count_list=[0 for j in range(10)]
        #all_label = set([k for k in range(10)])
        #label = set(labels[list(dict_users[i])])
        #zero_label = list(all_label - label)
        #if len(zero_label) > 0:
        #    for element in zero_label:
        #        count_list[element] = 0
        label_number = Counter(labels[list(dict_users[j])])
        for key,value in label_number.items():
            count_list[key] = value
        count_lists.append(count_list)

    #print(count_lists)




    return dict_users,count_lists




def cifar10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    count_lists = []
    labels = np.array(dataset.targets)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        #compute each label numbers
        count_list=[0 for j in range(10)]
        #all_label = set([k for k in range(10)])
        #label = set(labels[list(dict_users[i])])
        #zero_label = list(all_label - label)
        #if len(zero_label) > 0:
        #    for element in zero_label:
        #        count_list[element] = 0
        label_number = Counter(labels[list(dict_users[i])])
        for key,value in label_number.items():
            count_list[key] = value
        count_lists.append(count_list)

    return dict_users,count_lists



def cifar10_noniid_dirichlet(dataset, num_users, beta):
    #beta is smaller, daat is more un-balanced
    labels = np.array(dataset.targets)
    N = len(labels)
    dict_users = {}
    min_size = 0
    min_require_size = 10
    K = 10

    count_lists = []
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            #print(proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        count_list=[0 for j in range(10)]
        #all_label = set([k for k in range(10)])
        #label = set(labels[list(dict_users[i])])
        #zero_label = list(all_label - label)
        #if len(zero_label) > 0:
        #    for element in zero_label:
        #        count_list[element] = 0
        label_number = Counter(labels[list(dict_users[j])])
        for key,value in label_number.items():
            count_list[key] = value
        count_lists.append(count_list)

    #print(count_lists)




    return dict_users,count_lists



def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    count_lists = []
    labels = np.array(dataset.targets)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        #compute each label numbers
        count_list=[0 for j in range(100)]
        #all_label = set([k for k in range(10)])
        #label = set(labels[list(dict_users[i])])
        #zero_label = list(all_label - label)
        #if len(zero_label) > 0:
        #    for element in zero_label:
        #        count_list[element] = 0
        label_number = Counter(labels[list(dict_users[i])])
        for key,value in label_number.items():
            count_list[key] = value
        count_lists.append(count_list)

    return dict_users,count_lists


def cifar100_noniid_dirichlet(dataset, num_users, beta):
    #beta is smaller, daat is more un-balanced
    labels = np.array(dataset.targets)
    N = len(labels)
    dict_users = {}
    min_size = 0
    min_require_size = 10
    K = 100

    count_lists = []
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            #print(proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        count_list=[0 for j in range(100)]
        #all_label = set([k for k in range(10)])
        #label = set(labels[list(dict_users[i])])
        #zero_label = list(all_label - label)
        #if len(zero_label) > 0:
        #    for element in zero_label:
        #        count_list[element] = 0
        label_number = Counter(labels[list(dict_users[j])])
        for key,value in label_number.items():
            count_list[key] = value
        count_lists.append(count_list)

    #print(count_lists)




    return dict_users,count_lists




if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid_label(dataset_train, num)
