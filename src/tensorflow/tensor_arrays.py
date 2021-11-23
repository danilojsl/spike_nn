# import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    tensor1 = tf.constant([[1, 2, 3, 4, 5]])
    print(tensor1)
    reversed_tensor = tf.reverse(tensor1, [1])
    print(reversed_tensor)

    tensor2 = tf.constant([[6, 7, 8, 9, 10]])
    tensor_list = [tensor1, tensor2]
    print(tensor_list)
    concat_tensor = tf.concat(tensor_list, 0)
    print(concat_tensor)
