#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-27-21 01:06
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import string
import unicodedata
from io import open
import glob
import torch


def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"  # Five extra letters
n_letters = len(all_letters)


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


def letterToIndex(letter):
    # Find letter index from all_letters, e.g. "a" = 0
    return all_letters.find(letter)


def letterToIndex_test():
    for l in all_letters:
        index = all_letters.find(l)
        print(f"{l}: {index}; ", end='')


def letterToTensor(letter):
    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1  # One-hot encoding
    return tensor


def lineToTensor(line):
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def main():
    # Prepare data
    print(findFiles('data/names/*.txt'))
    print(unicodeToAscii('Ślusàrski'))


if __name__ == "__main__":
    main()
