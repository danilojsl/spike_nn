from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Activation
from numpy import array


def get_train():
    seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape((len(X), 1, 1))
    return X, y


if __name__ == '__main__':
    model = Sequential()
    forward_layer = LSTM(10, return_sequences=True)
    backward_layer = LSTM(10, activation='relu', return_sequences=True,
                          go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                            input_shape=(1, 1)))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # fit model
    X, y = get_train()
    model.fit(X, y, epochs=10, shuffle=False, verbose=0)

    # make predictions
    yhat = model.predict(X, verbose=0)
    print(yhat)
