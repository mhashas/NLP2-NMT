from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, hidden_size, input_size, max_pos, dropout = 0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.embeddings = nn.Embedding(input_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)

    def forward(self, input, positions, hidden, validation=False):
        input_embeddings = self.embeddings(input)
        positions_embeddings = self.position_embeddings(positions)

        if not validation:
            input_embeddings = F.dropout(input_embeddings, p=self.dropout_rate)
            positions_embeddings = F.dropout(positions_embeddings, p=self.dropout_rate)

        input = torch.add(input_embeddings, positions_embeddings)

        return torch.mean(input, dim=1).unsqueeze(0), input


    def eval(self, english, positions, hidden):
        return self.forward(english, positions, hidden, True)

    def init_hidden(self, batch_size):
        return Variable(torch.randn(2, batch_size, self.hidden_size))