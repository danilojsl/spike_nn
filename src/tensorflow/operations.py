import numpy as np


def concat_two_tensors():
    array1 = np.random.rand(1, 3)
    array2 = None
    arrays = [array1, array2]
    valid_l = [x for x in arrays if x is not None]
    print(valid_l)
    dimension = len(valid_l[0].shape) - 1
    result = np.concatenate(valid_l, dimension)
    print(result.shape)
    print(result)


def concat_three_tensors():
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


if __name__ == '__main__':
    concat_two_tensors()
