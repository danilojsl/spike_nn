import tensorflow as tf


def scalar_automatic_differentiation():
    print('Automatic Differentiation for Scalar')
    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        y = x**2

    # dy = 2x * dx
    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy())


def tensor_automatic_differentiation():
    print('Automatic Differentiation for Tensors')
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    random_tensor = tf.compat.v1.get_variable("random_tensor", shape=(3, 2),
                                              initializer=tf.contrib.layers.xavier_initializer())
    w = tf.Variable(random_tensor, name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]

    with tf.GradientTape(persistent=True) as tape:
        y = tf.matmul(x, w) + b
        loss = tf.reduce_sum(y)

    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print("Weight values before applying gradients")
    print(w)
    print(dl_dw)
    print("Bias values before applying gradients")
    print(b)
    print(dl_db)
    opt.apply_gradients([(dl_dw, w), (dl_db, b)])
    print("Weight values after applying gradients")
    print(w)
    print(dl_dw)
    print("Bias values after applying gradients")
    print(b)
    print(dl_db)


def tensor_automatic_differentiation_minimize():

    def compute_loss():
        y = tf.matmul(x, w) + b
        loss = tf.reduce_sum(y)  # tf.reduce_mean(y**2)
        return loss

    print('Automatic Differentiation for Tensors')
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    random_tensor = tf.compat.v1.get_variable("random_tensor", shape=(3, 2),
                                              initializer=tf.contrib.layers.xavier_initializer())
    w = tf.Variable(random_tensor, name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]

    print("Values before applying gradients")
    print(w)
    print(b)
    opt.minimize(compute_loss, [w, b])
    print("Values after applying gradients")
    print(w)
    print(b)


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    scalar_automatic_differentiation()
    tensor_automatic_differentiation_minimize()
