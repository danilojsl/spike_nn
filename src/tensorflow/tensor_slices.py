import tensorflow as tf

tensor = tf.constant([[[1, 2, 3], [4, 5, 6]],
                     [[7, 8, 9], [10, 11, 12]],
                     [[13, 14, 15], [16, 17, 18]]])

tensor2 = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])

if __name__ == '__main__':
    print("Extracting elements starting from top")
    print(tf.slice(tensor, begin=[0, 0, 0], size=[1, 1, 3]))
    print(tf.slice(tensor, begin=[0, 0, 0], size=[1, 1, -1]))
    print(tf.slice(tensor, begin=[0, 0, 0], size=[1, -1, -1]))
    print(tf.slice(tensor, begin=[0, 0, 0], size=[2, 1, 3]))
    print(tf.slice(tensor, begin=[0, 0, 0], size=[2, -1, -1]))
    print(tf.slice(tensor, begin=[0, 0, 0], size=[-1, -1, 3]))

    all_elements_second_dim = tf.slice(tensor, begin=[0, 0, 0], size=[-1, 2, -1])
    second_dim_elements = tf.reshape(all_elements_second_dim, shape=(1, -1))

    print(tf.slice(tensor, begin=[0, 0, 0], size=[1, 1, 3]))
    print(tf.slice(tensor, begin=[0, 1, 0], size=[1, 1, 3]))
    print(tf.slice(tensor, begin=[1, 0, 0], size=[1, 1, 3]))
    print(tf.slice(tensor, begin=[1, 1, 0], size=[1, 1, 3]))
    print(tf.slice(tensor, begin=[2, 0, 0], size=[1, 1, 3]))
    print(tf.slice(tensor, begin=[2, 1, 0], size=[1, 1, 3]))

    first_dim = tensor.shape.dims[0]
    second_dim = tensor.shape.dims[1]
    tensors = []
    for i in range(first_dim):
        for j in range(second_dim):
            sliced_tensor = tf.reshape(tf.slice(tensor, begin=[i, j, 0], size=[1, 1, -1]), shape=(1, -1))
            tensors.append(sliced_tensor)
    print(tensors)

    second_dim = tensor2.shape.dims[1]
    tensors2 = []
    for i in range(second_dim):
        sliced_tensor = tf.reshape(tf.slice(tensor2, begin=[0, i, 0], size=[1, 1, -1]), shape=(1, -1))
        tensors2.append(sliced_tensor)
    print(tensors2)
