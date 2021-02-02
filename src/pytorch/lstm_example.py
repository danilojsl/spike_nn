import torch
import torch.nn as nn

# Example of NN https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.number_features = 1
        hidden_state_size = 2
        self.sample_size = 1
        num_layers = 1

        self.initial_hidden_state = torch.zeros(num_layers, self.sample_size, hidden_state_size)  # h_0
        self.initial_cell_state = torch.zeros(num_layers, self.sample_size, hidden_state_size)  # c_0
        self.lstm1 = nn.LSTM(input_size=self.number_features, hidden_size=hidden_state_size)
        print(self.lstm1)

    def forward(self):
        time_steps = 3  # sequence length
        random_input = torch.rand((self.sample_size, time_steps))
        input_sequence = random_input.reshape(time_steps, self.sample_size, self.number_features)

        output_features, (hidden_state, cell_state) = self.lstm1(input_sequence, (self.initial_hidden_state,
                                                                                  self.initial_cell_state))
        print("Output Features (h_t) from last layer of LSTM: " + str(output_features))
        print("Hidden State for t=seq_len : " + str(hidden_state))
        print("Cell State for t=seq_len : " + str(cell_state))


if __name__ == '__main__':
    lstm_net = Net()
    lstm_net.forward()
