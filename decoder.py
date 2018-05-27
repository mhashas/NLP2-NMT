import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate = 0.5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(output_size, hidden_size)

        #self.network = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.network = nn.RNN(hidden_size, hidden_size, batch_first=True)

        self.attention = Attention(hidden_size)

        self.attention_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, output, hidden, validation=False):
        # Apply attention to English sentence
        context = self.attention(output, hidden, validation, self.dropout_rate).transpose(0, 1)
        hidden = self.attention_combined(torch.cat((hidden, context), dim=2))

        input = self.embedding(input).unsqueeze(1)

        output, hidden = self.network(input, hidden)

        output_over_vocab = self.out(output[:, 0, :])
        probs = self.softmax(output_over_vocab)
        return probs, hidden

    def eval(self, input, output, hidden):
        probs, hidden = self.forward(input, output, hidden, True)

        weights = self.attention.last_weights
        return probs, hidden, weights

    def init_hidden(self, batch_size, enable_cuda):
        if enable_cuda:
            return Variable(torch.randn(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.randn(1, batch_size, self.hidden_size))

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.normaliser = np.sqrt(hidden_size)
        # Save weights for visualisation
        self.last_weights = None

    def forward(self, english, hidden, validation, dropout_rate, v=None):
        if v is None:
            hidden = hidden.transpose(0, 1).transpose(1, 2)
        dotproduct = torch.bmm(english, hidden).squeeze(2)
        dotproduct = dotproduct / self.normaliser
        weights = self.softmax(dotproduct)
        if not validation:
            weights = F.dropout(weights, p=dropout_rate)
        self.last_weights = weights.data[0, :]
        if v is None:
            return torch.bmm(weights.unsqueeze(1), english)
        else:
            return torch.bmm(weights.unsqueeze(1), v)