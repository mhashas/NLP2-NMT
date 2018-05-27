import logging
from nltk.tokenize import word_tokenize
from bpe import learn_bpe, BPE
from copy import deepcopy
from dictionary import *
from utils import *
from preprocess import *


class Dataset(object):
    """Collects words and corresponding associations, preprocesses them."""

    def __init__(self, train_en_file, train_fr_file, batch_size, helper : Preprocess):
        self.batch_size = batch_size
        self.lines_e = []
        self.lines_f = []
        self.max_pos = 0

        self.lines_e, self.lines_f, self.bpe_e, self.bpe_f = helper.preprocess(train_en_file, train_fr_file, 1)
        self.dict_e = Dictionary(self.lines_e)
        self.dict_f = Dictionary(self.lines_f)

        self.longest_english = max([len(e) for e in self.lines_e])
        self.longest_french = max([len(f) for f in self.lines_f])

        self.vocab_size_e = len(self.dict_e.word2index)
        self.vocab_size_f = len(self.dict_f.word2index)

        # Create batches
        self.batches = get_batches(self)

    def load_data(self, pathl1, pathl2):
        lines_e = []
        lines_f = []
        with open(pathl1, 'r', encoding='utf8') as f_eng, open(pathl2, 'r', encoding='utf8') as f_fre:
            for line_e in f_eng:
                line_f = f_fre.readline()
                line_e = " ".join(self.prepare(line_e))
                line_f = " ".join(self.prepare(line_f))
                line_e = self.bpe_e.process_line(line_e).split()
                line_f = self.bpe_f.process_line(line_f).split()
                lines_e.append(line_e)
                lines_f.append(line_f)
        return lines_e, lines_f

    def word_positions(self, line):
        result = []
        pos = 1
        for word in line:
            result.append(pos)
            if pos > self.max_pos: self.max_pos = pos
            if not (len(word) > 2 and word[-2:] == '@@'): pos += 1
        return result

    def prepare(self, sequence):
        sequence = sequence.lower()
        return ['<s>'] + word_tokenize(sequence) + ['</s>']