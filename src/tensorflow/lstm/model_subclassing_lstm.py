import tensorflow as tf
from keras.layers import LSTM
from numpy import array


def get_train():
    seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape((len(X), 1, 1))
    return X, y


class LSTMModule(tf.keras.Model):

    def __init__(self, lstm_dims):
        super().__init__()
        self.lstm = LSTM(lstm_dims, return_sequences=True, return_state=True)

    def call(self, inputs):
        lstm_output = self.lstm(inputs)
        return lstm_output


if __name__ == '__main__':
    lstm_module = LSTMModule(10)
    lstm_module.compile(loss='mse', optimizer='adam')
    X, y = get_train()
    lstm_module(X)
    # lstm_module.save("./model")
    # Prediction
    y_hat = lstm_module.predict(X)
    print(y_hat)
