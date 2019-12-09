import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

'''
Simple Neural Network having 3 dense layer and relu activation function and Adam optimizer

'''

def SimpleNetwork(input_size, output_size, last_act, loss, lr):
    network = Sequential()
    network.add(Dense(256, activation='relu', input_shape=input_size))
    network.add(Dense(256, activation='relu'))
    network.add(Dense(output_size, activation=last_act))
    network.compile(loss=loss, optimizer=Adam(lr=lr))
    network.summary()
    return network
