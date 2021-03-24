#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-19-20 17:31
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

from __future__ import unicode_literals, print_function, division

import os
import unicodedata
import re


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            # Identify the word by current n_words linearly
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1  # Increase current n_words
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_languages(lang1, lang2, reverse=False, path="./data"):
    """read_languages
    Note: this function only initializes Lang class for each language argument, but not add sentences for them!

    Input:
        lang1:
        lang2:

    Output:

    """
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(os.path.join(path, f"{lang1}-{lang2}.txt"), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs, max_length):
    """filterPairs
    Input:
        pairs:
        max_length: max length of the input

    Output:
        filtered language pairs
    """
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepare_data(lang1, lang2, reverse=False, max_length=10, path="./data"):
    """prepare_data
    Output:
        input_lang: input lang Lang class instance
        output_lang: out lang Lang class instance
        pairs: filtered language sentence pairs.
    """
    input_lang, output_lang, pairs = read_languages(
        lang1, lang2, reverse, path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs
