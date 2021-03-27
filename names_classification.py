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
import torch
from NLP_from_scratch.utils.names.data import n_letters, all_categories, n_categories, letterToTensor, lineToTensor, categoryFromOutput, randomTrainingExample
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


def training():
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

    log_dir = "./assets/results/names/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "all_losses.pickle")
    with open(path, "wb") as f:
        pickle.dump(all_losses, f)
        print(f"Saved all_losses to: {path}")

    save_dir = "./assets/results/names/weights"
    os.makedirs(save_dir, exist_ok=True)
    weights_name = f"rnn_model-loss-{all_losses[-1]:.4f}.pt"
    weights_path = os.path.join(save_dir, weights_name)
    torch.save(rnn.state_dict(), weights_path)
    print(f"Save weights to: {weights_path}.")


def evaluate(rnn, line_tensor):
    # Just return an output given a line
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def reload_model():
    # Reload weights
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    save_dir = "./assets/results/names/weights"
    only_pt_files = [f for f in os.listdir(save_dir) if os.path.isfile(
        os.path.join(save_dir, f)) and f.endswith(".pt")]

    weights_name = only_pt_files[-1]
    weights_path = os.path.join(save_dir, weights_name)
    rnn.load_state_dict(torch.load(weights_path))
    print(f"Loaded weights from: {weights_path}.")

    return rnn


def evaluating():
    rnn = reload_model()

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def predicting():
    rnn = reload_model()

    def predict(rnn, input_line, n_predictions=3):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = evaluate(rnn, lineToTensor(input_line))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print(f"({value:.2f}) {all_categories[category_index]}")
                predictions.append([value, all_categories[category_index]])

    predict(rnn, 'Dovesky')
    predict(rnn, 'Jackson')
    predict(rnn, 'Satoshi')


def main():
    # training()
    # evaluating()
    predicting()


if __name__ == "__main__":
    main()
