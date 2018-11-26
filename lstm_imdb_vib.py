#! -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.datasets import imdb


class VIB(Layer):
    """变分信息瓶颈层
    """
    def __init__(self, lamb, **kwargs):
        self.lamb = lamb
        super(VIB, self).__init__(**kwargs)
    def call(self, inputs):
        z_mean, z_log_var = inputs
        u = K.random_normal(shape=K.shape(z_mean))
        kl_loss = - 0.5 * K.sum(K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), 0))
        self.add_loss(self.lamb * kl_loss)
        return z_mean + K.exp(z_log_var / 2) * u
    def compute_output_shape(self, input_shape):
        return input_shape[0]


max_features = 20000
maxlen = 80
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


seq_in = Input(shape=(maxlen,))
seq = seq_in

seq = Embedding(max_features, 128)(seq)
seq = LSTM(128)(seq)
z_mean = Dense(128)(seq)
z_log_var = Dense(128)(seq)
seq = VIB(0.001)([z_mean, z_log_var])
seq = Dense(1, activation='sigmoid')(seq)

model = Model(seq_in, seq)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))


"""
seq_in = Input(shape=(maxlen,))
seq = seq_in

seq = Embedding(max_features, 128)(seq)
seq = LSTM(128)(seq)
seq = Dense(1, activation='sigmoid')(seq)

model = Model(seq_in, seq)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
"""
