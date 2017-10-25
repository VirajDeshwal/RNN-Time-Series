import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    """
    Transforms the input series and window-size into a set of input/output
    pairs for use with our RNN model
    :param series: time series data that should be transformed
    :param window_size: size of the sliding window
    """
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size])

    y = series[window_size:]
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    P = len(series)-window_size
    y = np.asarray(y)
    y.shape = (len(y), 1)

    assert(type(X).__name__ == 'ndarray'), "X should be of type ndarray!"
    assert(type(y).__name__ == 'ndarray'), "y should be of type ndarray!"

    assert(X.shape == (P, window_size))
    assert(y.shape in [(P, 1), (P,)])

    return X, y


def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), activation='tanh'))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    punctuation = [' ', '!', ',', '.', ':', ';', '?']

    # print(len(' '*len(punctuation)))
    # transTab = text.maketrans(''.join(punctuation), ' '*len(punctuation))
    # text.translate(transTab)
    import string
    unique_ch = set(text)

    valid_chars = list(string.ascii_lowercase)
    valid_chars += punctuation
    invalid_chars = set(unique_ch).difference(valid_chars)

    for c in invalid_chars:
        text = text.replace(c, ' ')

    text.replace('  ', ' ')

    return text


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    P = len(text)
    M = step_size
    for i in range(0, P-window_size, M):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    return inputs, outputs


# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
