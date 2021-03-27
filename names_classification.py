#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-24-21 01:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division

import os
import time
import math
import pickle
from typing import no_type_check_decorator
import torch
import torch.nn as nn
from NLP_from_scratch.utils.names.data import n_letters, n_categories, letterToTensor, categoryFromOutput, randomTrainingExample
from NLP_from_scratch.models.names.rnn import RNN


def train(category_tensor, line_tensor, criterion, rnn, learning_rate):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    # Read line_tensor one letter (A <1 x n_letters> one-hot Tensor) by another
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    # Creating the Network
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    input = letterToTensor('A')
    hidden = torch.zeros(1, n_hidden)

    # Initial output
    output, next_hidden = rnn(input, hidden)
    print(f"output: {output}")
    print(f"next_hidden: {next_hidden}")
    print(categoryFromOutput(output))

    # Training
    criterion = nn.NLLLoss()
    # If you set this too high, it might explode. If too low, it might not learn
    learning_rate = 0.005

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(
            category_tensor, line_tensor, criterion, rnn, learning_rate)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print(
                f"{iter} {iter/n_iters*100}% ({timeSince(start)}) {loss:.4f} {line} / {guess} {correct}")

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    log_dir = "./assets/results/names"
    os.makedirs(log_dir, exist_ok=True)

    path = os.path.join(log_dir, "all_losses.pickle")
    with open(path, "wb") as f:
        pickle.dump(all_losses, f)
        print(f"Saved all_losses to: {path}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(all_losses)
    plt.show()


if __name__ == "__main__":
    main()
