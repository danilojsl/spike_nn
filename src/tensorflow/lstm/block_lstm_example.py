import tensorflow as tf

if __name__ == '__main__':
    input_sequence = tf.constant([[[0.1, 0.2]], [[0.3, 0.4]]])
    cell_size = 3
    ini_cell_state = tf.zeros(shape=[1, cell_size])
    ini_hidden_state = tf.zeros(shape=[1, cell_size])
    bias = tf.zeros(shape=[cell_size * 4])
    seq_len_max = tf.constant([2], dtype="int64")

    weight_matrix = tf.constant([
        [1.6652163, 1.366376, 0.7786316, 0.9834321, 1.6551187, -0.6363001, -0.4229284, 0.63195646, 0.6605189, -0.6906152, 3.1515226, 1.970373],
        [1.9458166, 0.9790728, 0.7476161, -1.6813406, -0.75150734, 0.13104685, 0.004470979, 0.009482844, -1.1464607, 0.5036645, 1.3567412, 0.71478313],
        [0.5393334, -0.6881541, 1.5186735, 1.3431606, -0.61521095, -2.1862414, 1.2603592, -0.33593372, -0.48804748, -0.34496853, -0.8777565, 0.9202126],
        [1.3439888, 0.32253885, -0.7401764, 0.10057431, -1.3759913, 0.08382488, 0.56741005, 2.207029, -0.0066946335, -0.8636334, 1.9623716, 0.14416508],
        [-0.925145, 0.2283957, 0.79638815, 0.2288384, 0.7052175, -0.18524477, -2.308545, 1.2240901, 2.014674, 0.6235778, -0.15852839, 0.17711076]
    ])

    weight_gates = tf.constant([1.6652163, 1.366376, 0.7786316])

    block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=seq_len_max, x=input_sequence, cs_prev=ini_cell_state,
                                      h_prev=ini_hidden_state, w=weight_matrix,
                                      wci=weight_gates, wcf=weight_gates,
                                      wco=weight_gates, b=bias)

    print(f"Input Gate: {block_lstm.i.numpy()}")
    print(f"Cell State: {block_lstm.cs.numpy()}")
    print(f"Forget State: {block_lstm.f.numpy()}")
    print(f"Output Gate: {block_lstm.o.numpy()}")
    print(f"Cell Input: {block_lstm.ci.numpy()}")
    print(f"Cell Output: {block_lstm.co}")
    print(f"Hidden Output: {block_lstm.h}")
