#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-28-21 01:32
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

import os
import time
import math
import pickle
import torch
import torch.nn as nn
from NLP_from_scratch.utils.names_generation import n_letters, all_categories, n_categories, randomTrainingExample
from NLP_from_scratch.models.names_generation import RNN


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(rnn, learning_rate, criterion, category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def training():
    # Creating the Network
    n_hidden = 128
    # n_letters -> n_letters
    rnn = RNN(n_letters, n_hidden, n_letters, n_categories)

    # Training
    criterion = nn.NLLLoss()
    learning_rate = 0.0005

    # Parameters
    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0  # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(rnn, learning_rate, criterion,
                             *randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0


def main():
    training()
    # evaluating()
    # predicting()


if __name__ == "__main__":
    main()
