import numpy as np
import tensorflow as tf


def concat_two_numpy():
    array1 = np.random.rand(1, 3)
    array2 = None
    arrays = [array1, array2]
    valid_l = [x for x in arrays if x is not None]
    print(valid_l)
    dimension = len(valid_l[0].shape) - 1
    result = np.concatenate(valid_l, dimension)
    print(result.shape)
    print(result)


def concat_three_numpy():
    array1 = np.random.rand(2, 3)
    array2 = np.random.rand(2, 3)
    array3 = None
    arrays = [array1, array2, array3]
    valid_l = [x for x in arrays if x is not None]
    print(valid_l)
    dimension = len(valid_l[0].shape) - 1
    result = np.concatenate(valid_l, dimension)
    print(result.shape)
    print(result)


def sum_tensors():
    tensor1 = tf.random.normal(shape=[2, 3])
    print(tensor1)
    tensor2 = tf.random.normal(shape=[2, 3])
    print(tensor2)

    plus_result = tensor1 + tensor2
    print(plus_result)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    sum_tensors()
