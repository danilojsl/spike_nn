import tensorflow as tf
import numpy as np


class CustomEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="random_normal",
            dtype="float32",
        )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    layer = CustomEmbedding(10, 32, mask_zero=True)
    x = np.random.random((3, 10)) * 9
    x = x.astype("int32")

    y = layer(x)
    mask = layer.compute_mask(x)

    print(mask)
