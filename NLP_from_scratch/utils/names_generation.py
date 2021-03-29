#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-29-21 18:37
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division

from io import open
import glob
import random
import os
import unicodedata
import string
import torch


def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'-"  # Six extra letters
n_letters = len(all_letters) + 1  # Plus EOS marker


def unicodeToAscii(s):
    """unicodeToAscii
    Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    For example, 'Ślusàrski' -> 'Slusarski'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    """
    Read a file and split into lines.
    Each file of 'data/names/*.txt' is a text file with names belonging to a category.
    Every line of these files is a name string belonging to the category of the file.
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingPair():
    """randomTrainingPair
    Get a random category and random line from that category.

    Output:
        category: a random category, e.g., "English".
        line: a random line of name belonging to above category.
    """
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


def categoryTensor(category):
    """categoryTensor
    Convert category to one-hot vector.

    Output: One-hot vector for category.
    """
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1

    return tensor


def inputTensor(line):
    """inputTensor
    Same function as `lineToTensor`

    Output: 
        One-hot matrix of first to last letters (not including EOS) for input
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def targetTensor(line):
    """targetTensor

    Output: 
        LongTensor/torch.int64 (with shape 1 x (len(line)+1) ) of second letter to end (EOS) for target
    """
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS's index
    return torch.LongTensor(letter_indexes)


def randomTrainingExample():
    """
    Make category, input, and target tensors from a random category, line pair
    """
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def main():
    pass


if __name__ == "__main__":
    main()
