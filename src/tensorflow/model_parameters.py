import tensorflow as tf


def define_parameter():
    random_tensor = tf.get_variable("random_tensor", shape=(5, 10),
                                    initializer=tf.contrib.layers.xavier_initializer())
    parameter = tf.Variable(random_tensor)
    print(parameter)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    define_parameter()
