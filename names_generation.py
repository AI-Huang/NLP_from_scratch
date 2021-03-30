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
from NLP_from_scratch.utils.names_generation import all_letters, n_letters, all_categories, n_categories
from NLP_from_scratch.utils.names_generation import randomTrainingExample, categoryTensor, inputTensor
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
            print(
                f"{timeSince(start)}|iter: ({iter} {int(iter/n_iters*100)}%)|loss: {loss:.4f}")

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    log_dir = "./assets/results/names_generation/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "all_losses.pickle")
    with open(path, "wb") as f:
        pickle.dump(all_losses, f)
        print(f"Saved all_losses to: {path}")

    save_dir = "./assets/results/names_generation/weights"
    os.makedirs(save_dir, exist_ok=True)
    weights_name = f"rnn_model-loss-{all_losses[-1]:.4f}.pt"
    weights_path = os.path.join(save_dir, weights_name)
    torch.save(rnn.state_dict(), weights_path)
    print(f"Save weights to: {weights_path}.")


def plotting():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    log_dir = "./assets/results/names_generation/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "all_losses.pickle")
    with open(path, "rb") as f:
        all_losses = pickle.load(f)

    plt.figure()
    plt.plot(all_losses)
    plt.title("Training loss")
    plt.xlabel("Steps'00")
    plt.ylabel("NIL loss")
    plt.grid()
    plt.show()


def restore_model():
    # Restore network and  weights
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_letters, n_categories)

    save_dir = "./assets/results/names_generation/weights"
    only_pt_files = [f for f in os.listdir(save_dir) if os.path.isfile(
        os.path.join(save_dir, f)) and f.endswith(".pt")]

    weights_name = only_pt_files[-1]
    weights_path = os.path.join(save_dir, weights_name)
    rnn.load_state_dict(torch.load(weights_path))
    print(f"Loaded weights from: {weights_path}.")

    return rnn


def sample(rnn, category, start_letter='A', max_length=20):
    """
    Sample from a category and starting letter.
    """
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


def samples(rnn, category, start_letters='ABC'):
    """samples
    Get multiple samples from one category and multiple starting letters
    """
    for start_letter in start_letters:
        print(sample(rnn, category, start_letter))


def sampling():
    # Restore network
    rnn = restore_model()

    # Sampling
    samples(rnn, 'Russian', 'RUS')
    samples(rnn, 'German', 'GER')
    samples(rnn, 'Spanish', 'SPA')
    samples(rnn, 'Chinese', 'CHI')


def main():
    # training()
    # plotting()
    sampling()
    # predicting()


if __name__ == "__main__":
    main()
