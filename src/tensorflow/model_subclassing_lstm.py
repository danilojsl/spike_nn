import tensorflow as tf
from tensorflow.keras.layers import LSTM


def init_hidden(dim):
    return tf.zeros(shape=[1, dim]), tf.zeros(shape=[1, dim])


class LSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super().__init__()
        # Input data
        self.lstm_dims = lstm_dims
        self.lstm = LSTM(lstm_dims, return_sequences=True, return_state=True)

    def call(self, inputs):
        # Forward pass
        ini_hidden_state = init_hidden(self.lstm_dims)
        hidden_states, (hidden_state, cell_state) = self.get_lstm_output(self.lstm, inputs, ini_hidden_state)

        return hidden_states, hidden_state, cell_state

    @staticmethod
    def get_lstm_output(lstm_model, input_sequence, initial_state):
        output = lstm_model(input_sequence, initial_state=initial_state)
        hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
        return hidden_states, (hidden_state, cell_state)


if __name__ == '__main__':
    lstm_module = LSTMModule(5)
    lstm_input = tf.constant([[0.1, 0.2], [0.3, 0.4]], shape=[1, 2, 2])
    lstm_output = lstm_module(lstm_input)
    print(lstm_output)
