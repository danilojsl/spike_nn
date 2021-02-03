from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
import numpy as np


class SimpleLstm:

    def __init__(self):
        # LSTM Network architecture
        self.time_steps = 3
        self.number_features = 1
        self.hidden_state_size = 2

        # Input data
        self.sample_size = 1  # batch size

    def forward(self):
        inputs1 = Input(shape=(self.time_steps, self.number_features))
        lstm1, hidden_states, cell_state = \
            LSTM(self.hidden_state_size, return_sequences=True, return_state=True)(inputs1)
        return Model(inputs=inputs1, outputs=[lstm1, hidden_states, cell_state])

    def prediction(self, lstm_model):
        lstm_model.summary()
        input_sequence = self.get_input()
        output = lstm_model.predict(input_sequence)

        hidden_states = output[0]
        hidden_state = output[1]
        cell_state = output[2]
        print("Hidden State for each input time step: " + str(hidden_states))
        print("Hidden State for last time step: " + str(hidden_state))
        print("Cell State for last time step: " + str(cell_state))

    def get_input(self):
        random_input = np.random.rand(self.sample_size, self.time_steps)
        input_sequence = random_input.reshape((self.sample_size, self.time_steps, self.number_features))
        return input_sequence


if __name__ == '__main__':
    lstm_net = SimpleLstm()
    simple_lstm_model = lstm_net.forward()
    lstm_net.prediction(simple_lstm_model)
