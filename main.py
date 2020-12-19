#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-18-20 20:18
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import random
import torch
from utils.load_data import prepare_data
from torch_fn.seq2seq import EncoderRNN, AttnDecoderRNN
from torch_fn.training import tensors_from_pair, train_iterations

MAX_LENGTH = 10


def main():
    input_lang, output_lang, pairs = prepare_data(
        'eng', 'fra', True, max_length=MAX_LENGTH)
    print(random.choice(pairs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(
        hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)

    n_iters = 75000
    training_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs), device=device)
                      for _ in range(n_iters)]

    train_iterations(training_pairs, encoder1, attn_decoder1, n_iters=n_iters,
                     print_every=5000, max_length=MAX_LENGTH, device=device)


if __name__ == "__main__":
    main()
