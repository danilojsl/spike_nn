from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import numpy as np


class SimpleLstm:

    def __init__(self):
        # LSTM Network architecture
        self.time_steps = 3
        self.number_features = 2
        self.hidden_state_size = 4

        # Input data
        self.sample_size = 1  # batch size

    def forward(self):
        inputs1 = Input(shape=(self.time_steps, self.number_features))
        hidden_states, hidden_state, cell_state = \
            LSTM(self.hidden_state_size, return_sequences=True, return_state=True)(inputs1)
        return Model(inputs=inputs1, outputs=[hidden_states, hidden_state, cell_state])

    def prediction(self, lstm_model):
        lstm_model.summary()
        input_sequence = self.__get_tensor_input()
        output = lstm_model.predict(input_sequence)

        hidden_states = output[0]
        hidden_state = output[1]
        cell_state = output[2]
        print("Hidden State for each input time step: " + str(hidden_states))
        print("Hidden State for last time step: " + str(hidden_state))
        print("Cell State for last time step: " + str(cell_state))

    def __get_numpy_input(self):
        # Since in Pytorch is [14, 1, 100] in Tensorflow should be [1, 14, 100]
        random_input = np.random.rand(self.time_steps, self.number_features)
        input_sequence = random_input.reshape((self.sample_size, self.time_steps, self.number_features))
        print("Input Sequence: " + str(input_sequence))
        return input_sequence

    def __get_tensor_input(self):
        random_input = tf.random.normal(shape=[self.time_steps, self.number_features])
        input_sequence = tf.reshape(random_input, [self.sample_size, self.time_steps, self.number_features])
        print("Input Sequence: " + str(input_sequence))
        return input_sequence


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()  # to debug tf.tensor values
    lstm_net = SimpleLstm()
    simple_lstm_model = lstm_net.forward()
    lstm_net.prediction(simple_lstm_model)
