import logging
from bpe import bpe_to_sentence
from encoder import *
from decoder import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tqdm
from utils import *

class Model:

    def __init__(self, hidden_size, corpus):
        self.encoder = Encoder(hidden_size, corpus.vocab_size_e, corpus.max_pos)
        self.decoder = Decoder(hidden_size, corpus.vocab_size_f)

    def train(self, corpus, validation, testing, lr, epochs, batch_size, ratio, max_length):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()), lr=lr)
        losses = []
        bleus = []
        for i in range(epochs):
            epoch_loss = 0

            for input, input_positions, output, _ in tqdm.tqdm(corpus.batches):
                optimizer.zero_grad()
                use_teacher_forcing = True if random.random() < ratio else False

                encoder_hidden_state = self.encoder.init_hidden(batch_size)
                decoder_hidden_state, input = self.encoder(input, input_positions, encoder_hidden_state)


                output_token = output[:, 0]
                loss = 0
                for j in range(1, output.shape[1]):
                    probs, decoder_hidden_state = self.decoder(output_token, input, decoder_hidden_state)
                    loss += criterion(probs, output[:, j])
                    if use_teacher_forcing:
                        output_token = output[:, j]
                    else:
                        _, output_token = torch.topk(probs, 1)
                        output_token = output_token[:, 0]

                loss.backward()
                optimizer.step()
                epoch_loss += loss.data[0] / output.shape[1]

            bleus.append(self.evaluate(corpus, validation, max_length))

            epoch_loss = epoch_loss / len(corpus.batches)
            logging.info("Loss per token: {}".format(epoch_loss))
            losses.append(epoch_loss)

        #bleus.append(self.evaluate(corpus, testing, max_length))
        return losses, bleus

    def evaluate(self, corpus, valid, max_length):

        scores = []

        for input, output in tqdm.tqdm(list(zip(valid[0], valid[1]))):
            positions = corpus.word_positions(input)
            indices = to_indices(input, corpus.dict_e)
            translation = greedy(self.encoder, self.decoder, indices, positions, corpus.dict_f.word2index, corpus.dict_f.index2word, max_length)

            output = bpe_to_sentence(output)
            translation = bpe_to_sentence(translation)
            scores.append(sentence_bleu([output], translation, smoothing_function=SmoothingFunction().method1))

        score = sum(scores) / len(scores)
        print("Greedy average BLEU score: {}".format(score))

        return score

