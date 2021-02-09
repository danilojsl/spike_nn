import tensorflow as tf


def simple_tensor_operation():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    print(product)


if __name__ == '__main__':
    # with tf.Session() as sess:
    tf.compat.v1.enable_eager_execution()
    simple_tensor_operation()
