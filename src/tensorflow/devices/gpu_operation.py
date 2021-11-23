import tensorflow as tf

if __name__ == '__main__':

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
        with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):       # Run nodes with GPU 0
            m1 = tf.constant([[3, 5]])
        m2 = tf.constant([[2], [4]])
        product = tf.matmul(m1, m2)
        print(sess.run(product))

