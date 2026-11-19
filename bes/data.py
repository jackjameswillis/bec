"""MNIST data pipeline shared by the STE baseline and BES.

Mirrors ``/data/ste/train_mnist_bnn.py`` exactly: flatten to 784, standardise
with the full training-set mean/std (+1e-7), one-hot labels. The last 10% of
the training set is split off deterministically as a validation set (identical
to Keras ``validation_split=0.1`` semantics), so BES evaluates fitness
on exactly the 54000 samples the STE baseline trains on.
"""

import tensorflow as tf

INPUT_DIM = 784
NUM_CLASSES = 10
N_VAL = 6000


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, INPUT_DIM).astype("float32")
    x_test = x_test.reshape(-1, INPUT_DIM).astype("float32")

    mean = x_train.mean()
    std = x_train.std()

    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    x_fit, x_val = x_train[:-N_VAL], x_train[-N_VAL:]
    y_fit, y_val = y_train[:-N_VAL], y_train[-N_VAL:]
    return (x_fit, y_fit), (x_val, y_val), (x_test, y_test)
