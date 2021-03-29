#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-28-21 23:50
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size +
                             hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size +
                             hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def main():
    n_letters, n_categories = 56, 18
    n_hidden = 128
    # Generating names, n_letters -> n_letters
    rnn = RNN(n_letters, n_hidden, n_letters, n_categories)


if __name__ == "__main__":
    main()
