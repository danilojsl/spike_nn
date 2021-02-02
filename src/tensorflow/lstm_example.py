from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
import numpy as np


def simple_lstm():
    """
    Define Model:
    Very small model with a single LSTM layer that itself contains a LSTM cell of size 2 (hidden state of length 2).
    In this example, we will have one input sample with 3 time steps and one feature observed at each time step
    Check book page 52 for parameters
    """
    time_steps = 3
    hidden_state_size = 2
    number_features = 2

    inputs1 = Input(shape=(time_steps, number_features))
    lstm1, hidden_states, cell_state = LSTM(hidden_state_size, return_sequences=True, return_state=True)(inputs1)
    model = Model(inputs=inputs1, outputs=[lstm1, hidden_states, cell_state])

    # define input data
    sample_size = 1  # batch size
    random_input = np.random.rand(sample_size, time_steps)
    input_sequence = random_input.reshape((sample_size, time_steps, number_features))

    # make and show prediction
    model.summary()
    output = model.predict(input_sequence)
    hidden_states = output[0]
    hidden_state = output[1]
    cell_state = output[2]
    print("Hidden State for each input time step: " + str(hidden_states))
    print("Hidden State for last time step: " + str(hidden_state))
    print("Cell State for last time step: " + str(cell_state))


if __name__ == '__main__':
    simple_lstm()
