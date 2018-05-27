
import logging
from encoder import *
from decoder import *
from dictionary import *
from dataset import Dataset
from model import *
from constants import *
from preprocess import *

BATCH_SIZE = 32
BPE_NUM_SYMBOLS = 10000
BPE_MIN_COUNT = 1
MAX_LENGTH = 74
TF_RATIO = 0.75
LEARNING_RATE = 0.001
EPOCHS = 10
REMOVE_PUNCTUATION = 1


helper = Preprocess(REMOVE_PUNCTUATION, BPE_NUM_SYMBOLS, BPE_MIN_COUNT)
dataset = Dataset(TRAIN_EN_FILE, TRAIN_FR_FILE, BATCH_SIZE, helper)

valid_en, valid_fr, _, _ = helper.preprocess(VAL_EN_FILE, VAL_FR_FILE, dataset.bpe_e, dataset.bpe_f)
validation = [valid_en, valid_fr]

test_en, test_fr, _, _ = helper.preprocess(TEST_EN_FILE, TEST_FR_FILE, dataset.bpe_e, dataset.bpe_f)
test = [test_en, test_fr]

hidden_sizes = [200]
for hidden_size in hidden_sizes:
    model = Model(hidden_size, dataset)
    losses, bleus = model.train(dataset, validation, test, LEARNING_RATE, EPOCHS, BATCH_SIZE, TF_RATIO, MAX_LENGTH)