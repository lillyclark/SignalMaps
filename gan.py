from __future__ import print_function, division
from signalmaps_helper import *

import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():

    def tf_make_gridpoints(self):
            testpoints = np.mgrid[self.x1min:self.x1max:30j, self.x2min:self.x2max:30j].reshape(2,-1).T
            return tf.constant(testpoints, dtype=tf.float32)

    def tf_vandermonde(self, x, num_features):
        if x.shape[1] > 2:
            i, j = 13, 12
        else:
            i, j = 0, 1
        return tf.transpose(tf.stack([tf.ones_like(x[:,i]),
                                      x[:,i],
                                      x[:,j],
                                      tf.square(x[:,i]),
                                      tf.square(x[:,j]),
                                      tf.multiply(x[:,i], x[:,j]),
                                      tf.multiply(tf.square(x[:,i]), x[:,i]),
                                      tf.multiply(tf.square(x[:,j]), x[:,j]),
                                      tf.multiply(tf.square(x[:,i]), x[:,j]),
                                      tf.multiply(tf.square(x[:,j]), x[:,i]),
                                      tf.multiply(tf.square(x[:,i]), tf.square(x[:,i])),
                                      tf.multiply(tf.square(x[:,j]), tf.square(x[:,j])),
                                      tf.multiply(tf.square(x[:,i]), tf.square(x[:,j])),
                                      tf.multiply(tf.square(x[:,i]), tf.multiply(x[:,i], x[:,j])),
                                      tf.multiply(tf.square(x[:,j]), tf.multiply(x[:,j], x[:,i]))
                                      ]))

    def tf_polynomial(self, x, num_features):
        if x.shape[1] > 2:
            i, j = 13, 12
        else:
            i, j = 0, 1
        return tf.transpose(tf.stack([tf.ones_like(x[:,i]),
                                      x[:,i],
                                      x[:,j],
                                      tf.square(x[:,i]),
                                      tf.square(x[:,j]),
                                      tf.multiply(x[:,i], x[:,j])
                                      ]))

    def tf_beta(self, vandermonde_x, rssinv):
        beta = tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(vandermonde_x), vandermonde_x)),
                              tf.matmul(tf.transpose(vandermonde_x), tf.reshape(rssinv,[rssinv.shape[0],1])))
        return beta

    def tf_fitmap(self, data, num_degrees, show=False, cell_tower=None):
        # v = self.tf_vandermonde(data, num_degrees)
        v = self.tf_polynomial(data, num_degrees)
        b = self.tf_beta(v, data[:,6])
        grid = self.tf_make_gridpoints()
        # v_grid = self.tf_vandermonde(grid, num_degrees)
        v_grid = self.tf_polynomial(grid, num_degrees)
        rss_pred = tf.matmul(v_grid, b)
        return v, b, grid, v_grid, rss_pred

    def __init__(self, all_data, norm_all_data, rho, batch_size):
        self.rho = rho
        self.batch_size = batch_size
        self.all_data = all_data
        self.norm_all_data = norm_all_data
        self.input_shape = (batch_size, norm_all_data.shape[1]-1)
        self.x1min, self.x1max, self.x2min, self.x2max = np.min(norm_all_data[:,13]), np.max(norm_all_data[:,13]), np.min(norm_all_data[:,12]), np.max(norm_all_data[:,12])

        # TODO
        optimizer = Adam(lr=0.001, beta_1=0.9)

        self.adversary = self.build_adversary()
        self.adversary.compile(loss=self.adversary_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        self.privatizer = self.build_privatizer()

        x = Input(shape=self.input_shape)
        self.x = x
        y = self.privatizer(x)
        self.y = y

        self.adversary.trainable = False

        uhat = self.adversary(y)

        self.combined = Model(x, uhat)
        self.combined.compile(loss=self.privatizer_loss,
                              optimizer=optimizer)


    def privatizer_loss(self, u, uhat):

        polynomial_degree = 2
        vx, bx, gridx, v_gridx, rssinv_predx = self.tf_fitmap(self.x[0], polynomial_degree)
        vy, by, gridy, v_gridy, rssinv_predy = self.tf_fitmap(self.y[0], polynomial_degree)
        utility_loss = tf.reduce_mean(tf.square(by-bx))
        privacy_loss = -1*keras.losses.categorical_crossentropy(u, uhat)
        return self.rho*utility_loss + (1-self.rho)*privacy_loss

    def adversary_loss(self, u, uhat):
        return keras.losses.categorical_crossentropy(u, uhat)

    def build_privatizer(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.input_shape[1]))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.input_shape[1]))

        # model.summary()

        x = Input(shape=self.input_shape)
        y = model(x)

        return Model(x, y)

    def build_adversary(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.input_shape[1]))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(5, activation='softmax'))

        # model.summary()

        y = Input(shape=self.input_shape)
        uhat = model(y)

        return Model(y, uhat)

    def train(self, epochs, adversary_epochs, seed=False):

        # load dataset
        X_train = self.norm_all_data[:,:25]
        print(X_train.shape)
        u = to_categorical(self.all_data[:,25])
        print(u.shape)

        for epoch in range(epochs):

            # Select a random batch
            if seed:
                np.random.seed(0)
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            X_train_batch = X_train[idx].reshape(1, self.batch_size, 25)
            self.x = X_train_batch
            u_train_batch = u[idx].reshape(1, self.batch_size, 5)

            # generate obfuscated data
            Y_batch = self.privatizer.predict(X_train_batch)
            self.y = Y_batch

            # Train the adversary
            self.adversary.trainable = True
            for adversary_epoch in range(adversary_epochs):
                self.a_loss = self.adversary.train_on_batch(Y_batch, u_train_batch)
            self.adversary.trainable = False

            # generate adversary estimates
            uhat_batch = self.adversary.predict(Y_batch)

            # Train the privatizer
            self.p_loss = self.combined.train_on_batch(X_train_batch, u_train_batch)

            # log the progress
            if epoch % 10 == 0:
                print ("%d [A loss: %f, acc.: %.2f%%] [P loss: %f]" % (epoch, self.a_loss[0], 100*self.a_loss[1], self.p_loss))

    def eval_privatizer(self, x):
        y = self.privatizer.predict(x)
        print("Input Data")
        vx, bx, gridx, v_gridx, rssinv_predx = fitmap(x.reshape((self.batch_size, 25)), 4, show=True)
        print("Obbfuscated Data")
        vy, by, gridy, v_gridy, rssinv_predy = fitmap(y.reshape((self.batch_size, 25)), 4, show=True)
        return y

all_data = np.genfromtxt('alldata')
norm_all_data_not_inverse = np.genfromtxt('norm_all_data_not_inverse')

print("setting up GAP")
# rho of 1 is all utility, rho 0 is all privacy
gan = GAN(all_data, norm_all_data_not_inverse, rho=0.0, batch_size=128)

print("training")
gan.train(epochs=100, adversary_epochs = 1, seed=False)

# gan.eval_privatizer(gan.x)
