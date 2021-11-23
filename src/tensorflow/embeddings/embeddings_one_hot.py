import tensorflow as tf
import numpy as np

vocab = {
    "never": 0,
    "a": 1,
    "good": 2,
    "world": 3,
    "am": 4,
    "bye": 5,
    "now": 6,
    "cat": 7,
    "hat": 8,
    "or": 9
}

VOCAB_SIZE = len(vocab)
EMBED_SIZE = 5


if __name__ == '__main__':
    embedding_table = np.random.randint(0, 10, size=(VOCAB_SIZE, EMBED_SIZE))
    embedding_table = tf.convert_to_tensor(embedding_table, tf.float32)
    print(embedding_table)

    texts = [
        "now or never",
        "good bye world"
    ]
    for i, text in enumerate(texts, 1):
        print(f"text {i}: {text}")

    tokenized_texts = [text.split(' ') for text in texts]
    for i, tokens in enumerate(tokenized_texts, 1):
        print(f"text {i}: {tokens}")

    word_ids = [[vocab[word] for word in tokenized_text]
                for tokenized_text in tokenized_texts]

    for i, tokens in enumerate(word_ids, 1):
        print(f"text {i}: {tokens}")

    word_ids = tf.convert_to_tensor(word_ids)
    print(word_ids)

    ohe_ids = tf.one_hot(word_ids, VOCAB_SIZE)
    print(ohe_ids)

    text_embded_manual = tf.matmul(ohe_ids, embedding_table)
    print(text_embded_manual)

    # just verify for text 1
    expected_output = text_embded_manual[0]

    text1_ids = word_ids[0]

    # step 1: do the lookup against the embedding table for each id
    actual_output = [embedding_table[id] for id in text1_ids]

    # step 2: add a row dimension
    actual_output = [tf.expand_dims(emb, 0) for emb in actual_output]

    # step 3: concatenate at the row dimension
    actual_output = tf.concat(actual_output, axis=0)
    print(actual_output)
    # verify
    assert tf.experimental.numpy.array_equal(expected_output, actual_output)
