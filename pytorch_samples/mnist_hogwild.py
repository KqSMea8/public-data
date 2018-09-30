#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多进程训练
https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from simplecnn import SimpleCNN
from train import train


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    args = parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(args.seed)

    model = SimpleCNN()
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
