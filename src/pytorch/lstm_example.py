import torch
import torch.nn as nn

# Example of NN https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # LSTM Network architecture
        self.number_features = 1
        self.hidden_state_size = 2
        self.num_layers = 1

        # Input data
        self.sample_size = 1
        self.time_steps = 3  # sequence length

    def forward(self):
        return nn.LSTM(input_size=self.number_features, hidden_size=self.hidden_state_size)

    def prediction(self, lstm_model):
        input_sequence = self.get_input()

        initial_hidden_state = torch.zeros(self.num_layers, self.sample_size, self.hidden_state_size)  # h_0
        initial_cell_state = torch.zeros(self.num_layers, self.sample_size, self.hidden_state_size)  # c_0
        output_features, (hidden_state, cell_state) = lstm_model(input_sequence,
                                                                 (initial_hidden_state, initial_cell_state))

        print("Output Features (h_t) from last layer of LSTM: " + str(output_features))
        print("Hidden State for t=seq_len : " + str(hidden_state))
        print("Cell State for t=seq_len : " + str(cell_state))

    def get_input(self):
        random_input = torch.rand((self.sample_size, self.time_steps))
        input_sequence = random_input.reshape(self.time_steps, self.sample_size, self.number_features)
        return input_sequence


if __name__ == '__main__':
    lstm_net = Net()
    simple_lstm_model = lstm_net.forward()
    lstm_net.prediction(simple_lstm_model)
