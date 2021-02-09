import torch
import torch.nn as nn


def scalar(f):
    if type(f) == int:
        return torch.LongTensor([f])
    if type(f) == float:
        return torch.FloatTensor([f])


class SimpleEmbedding(nn.Module):

    def __init__(self, enum_word, wdims):
        super(SimpleEmbedding, self).__init__()

        self.lookup = nn.Embedding(len(enum_word) + 3, wdims)
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.vocab['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2

    def forward(self):
        entry_norm = "*root*"
        index = scalar(int(self.vocab.get(entry_norm, 0)))
        wordvec = self.lookup(index)
        print(wordvec)


if __name__ == '__main__':
    enum_word = {'*root*': 0, 'al': 1, '-': 2, 'zaman': 3, 'american': 5, 'forces': 6, 'killed': 7, 'shaikh': 8}
    wdims = 20
    simple_embedding = SimpleEmbedding(enum_word, wdims)
    simple_embedding.forward()
