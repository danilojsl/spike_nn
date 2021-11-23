from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
import numpy as np
import tensorflow as tf


class SimpleEmbedding:

    def __init__(self, enum_word, wdims):
        self.lookup = Sequential(Embedding(len(enum_word) + 3, wdims))
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

    def get_embeddings(self):
        entry_norm = "*root*"
        index = np.array(self.vocab.get(entry_norm, 0)).reshape(1)
        wordvec = self.lookup.predict(index)
        print(wordvec)


def process_simple_embedding():
    simple_embedding = SimpleEmbedding(enum_word, wdims)
    simple_embedding.get_embeddings()


@tf.function
def process_embeddings_with_tensors():
    embeddings_dims = 100
    vocab_size = 125
    word_lookup = Embedding(vocab_size, embeddings_dims, name='embedding_vocab',
                            embeddings_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))
    embeddings_input = np.array([3])
    word_vec = word_lookup(embeddings_input)
    print(word_vec)


if __name__ == '__main__':
    word_ids = [1, 2, 3]
    VOCAB_SIZE = 125
    result = tf.one_hot(word_ids, VOCAB_SIZE)
    enum_word = {'*root*': 0, 'al': 1, '-': 2, 'zaman': 3, 'american': 5, 'forces': 6, 'killed': 7, 'shaikh': 8}
    wdims = 20
    process_embeddings_with_tensors()
    # process_simple_embedding()
