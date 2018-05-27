from constants import *
import os
import nltk
from dictionary import *
from bpe import learn_bpe, BPE
from copy import deepcopy

class Preprocess:
    train_data = []
    test_data = []
    val_data = []

    remove_punctuation = None
    bpe_num_symbols = None
    bpe_min_count = None

    bpe_f = None
    bpe_e = None

    BOS = '<s>'
    EOS = '</s>'

    def __init__(self, remove_punctuation, bpe_num_symbols, bpe_min_count):
        self.remove_punctuation = remove_punctuation
        self.bpe_num_symbols = bpe_num_symbols
        self.bpe_min_count = bpe_min_count

    def checkForPunkt(self):
        nltk.download('punkt')

    def preprocess(self, en_path, fr_path, bpe_e = None, bpe_f = None, apply_bpe = 1):
        self.checkForPunkt()

        en_sentences = open(en_path, 'r').read().splitlines()
        fr_sentences = open(fr_path, 'r').read().splitlines()

        new_en_sentences = []
        new_fr_sentences = []

        for index, en_sentence in enumerate(en_sentences):
            fr_sentence = fr_sentences[index]

            '''
            if self.remove_punctuation:
                en_sentence = en_sentence.replace('?', '').replace('!', '').replace('.', '').replace(',', '')
                fr_sentence = fr_sentence.replace('?', '').replace('!', '').replace('.', '').replace(',', '')
            '''
            fr_sentence = self.tokenize(fr_sentence)
            en_sentence = self.tokenize(en_sentence)

            new_en_sentences.append(en_sentence)
            new_fr_sentences.append(fr_sentence)

        if apply_bpe:
            return self.apply_bpe(new_en_sentences, new_fr_sentences, bpe_e, bpe_f)

        return new_en_sentences, new_fr_sentences

    def apply_bpe(self, en_sentences, fr_sentences, bpe_e = None, bpe_f = None):
        if not bpe_e or not bpe_f:
            bpe_f, bpe_e = self.load_bpe(en_sentences, fr_sentences)

        new_en_sentences = []
        new_fr_sentences = []

        for index, en_sentence in enumerate(en_sentences):
            fr_sentence = fr_sentences[index]

            en_sentence = bpe_e.process_line(" ".join(en_sentence)).split()
            fr_sentence = bpe_f.process_line(" ".join(fr_sentence)).split()

            new_en_sentences.append(en_sentence)
            new_fr_sentences.append(fr_sentence)

        return new_en_sentences, new_fr_sentences, bpe_e, bpe_f


    def load_bpe(self, en_sentences, fr_sentences):
        english_dict = Dictionary(en_sentences)
        french_dict = Dictionary(fr_sentences)

        if not os.path.isfile(BPE_EN_FILE) and not os.path.isfile(BPE_FR_FILE):
            fd = open(BPE_EN_FILE, 'w', encoding='utf8')
            learn_bpe(english_dict.counts, fd, self.bpe_num_symbols, min_frequency=self.bpe_min_count)

            fd = open(BPE_FR_FILE, 'w', encoding='utf8')
            learn_bpe(french_dict.counts, fd, self.bpe_num_symbols, min_frequency=self.bpe_min_count)

        fd = open(BPE_FR_FILE, 'r', encoding='utf8')
        bpe_f = BPE(fd, vocab=deepcopy(french_dict.counts))

        fd = open(BPE_EN_FILE, 'r', encoding='utf8')
        bpe_e = BPE(fd, vocab=deepcopy(english_dict.counts))

        return bpe_f, bpe_e

    def tokenize(self, sentence):
        sentence = sentence.lower()
        return [self.BOS] + nltk.word_tokenize(sentence) + [self.EOS]


