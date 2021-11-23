import tensorflow as tf

EMBEDDINGS_SIZE = 3

vocab = {
    "*PAD*": 0,
    "never": 1,
    "a": 2,
    "good": 3,
    "world": 4,
    "am": 5,
    "bye": 6,
    "now": 7,
    "cat": 8,
    "hat": 9,
    "or": 10
}


def get_tokenized_text():
    text = 'My cat is a great cat'
    tokens = text.lower().split()
    return tokens


def get_word_ids():
    return list(vocab.values())


if __name__ == '__main__':
    tf.raw_ops.LookupTableFind
    word_ids = get_word_ids()
    # Begins Graph Session
    tf.compat.v1.disable_eager_execution()
    inputs = tf.compat.v1.placeholder(tf.int32, [None], name='word_ids')

    # This is where the embedding vectors live
    # This will be modified by the optimization unless trainable=False
    # I choose random normal distribution but you can try other distributions
    embeddings = tf.compat.v1.random_normal(shape=(len(vocab), EMBEDDINGS_SIZE))

    # this will return the embedding lookup
    embedded = tf.nn.embedding_lookup(embeddings, inputs)
    sentence_embedding = []
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        embeddings_table = session.run(embedded, {inputs: word_ids})
        print("Embedding Table")
        print(embeddings_table)

        print("Lookup word id 0 and 2")
        output = session.run(embedded, {inputs: [0, 2]})
        print(output)

        print("Look up word cat")
        word_id = vocab.get("cat")
        output = session.run(embedded, {inputs: [word_id]})
        print(output)

        print("Look up sentence")
        word_ids.clear()
        for token in get_tokenized_text():
            word_ids.append(vocab.get(token, 0))
        sentence_embedding = session.run(embedded, {inputs: word_ids})

        session.close()

    print(sentence_embedding)
