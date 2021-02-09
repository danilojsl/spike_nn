from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
import numpy as np


class SimpleEmbedding:

    def __init__(self, enum_word, wdims):
        self.lookup = Sequential(Embedding(len(enum_word) + 3, wdims))
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

    def forward(self):
        entry_norm = "*root*"
        index = np.array(self.vocab.get(entry_norm, 0)).reshape(1)
        wordvec = self.lookup.predict(index)
        print(wordvec)


if __name__ == '__main__':
    enum_word = {'*root*': 0, 'al': 1, '-': 2, 'zaman': 3, 'american': 5, 'forces': 6, 'killed': 7, 'shaikh': 8}
    wdims = 20
    simple_embedding = SimpleEmbedding(enum_word, wdims)
    simple_embedding.forward()
