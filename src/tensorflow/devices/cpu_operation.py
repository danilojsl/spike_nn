import tensorflow as tf

if __name__ == '__main__':
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
        # Construct 2 op nodes (m1, m2) representing 2 matrix.
        m1 = tf.constant([[3, 5]])
        m2 = tf.constant([[2], [4]])

        product = tf.matmul(m1, m2)    # A matrix multiplication op node

        print(sess.run(fetches=product))

    # sess.close()


