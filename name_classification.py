#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-24-21 01:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division
import random


import os
import torch
from NLP_from_scratch.utils.names.data import RNN
from NLP_from_scratch.models.names.rnn import RNN


def categoryFromOutput(output, all_categories):
    print(output.size())
    top_n, top_i = output.topk(1)
    print(top_i)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor(
        [all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


def main():
    # Prepare data
    print(findFiles('data/names/*.txt'))
    print(unicodeToAscii('Ślusàrski'))

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print(n_categories)

    print(category_lines['Italian'][:5])

    # Turning Names into Tensors
    # A letter is turned into a <1 x n_letters> one-hot Tensor
    print(letterToTensor('J'))
    # A name string is turned into a <line_length x 1 x n_letters> one-hot matrix, or an array of one-hot letter vectors
    # This will make the encoded input matrix very sparse.
    # print(lineToTensor('Jones'))
    print(lineToTensor('Jones').size())

    # Creating the Network
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    input = letterToTensor('A')
    hidden = torch.zeros(1, n_hidden)

    output, next_hidden = rnn(input, hidden)
    print(f"output: {output}")
    print(f"next_hidden: {next_hidden}")

    print(categoryFromOutput(output, all_categories))


if __name__ == "__main__":
    main()
