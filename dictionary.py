from collections import defaultdict, Counter


class Dictionary(object):
    def __init__(self, sentences = None):
        self.word2index = defaultdict(lambda: len(self.word2index))
        self.index2word = dict()
        self.counts = Counter()

        if sentences:
            for sentence in sentences:
                self.add_sentence(sentence)

    def add_word(self, word):
        index = self.word2index[word]
        self.index2word[index] = word
        self.counts[word] += 1
        return index

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)
