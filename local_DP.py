from __future__ import print_function, division
from signalmaps_helper import *

import math
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

class NOISE_PRIVATIZER():

    def __init__(self, all_data, norm_all_data, num_users, batch_size, epsilon, delta, norm_clip):
        self.num_users = num_users
        self.epsilon = epsilon
        self.delta = delta
        self.norm_clip = norm_clip
        self.batch_size = batch_size
        self.all_data = all_data
        self.norm_all_data = norm_all_data
        self.input_shape = (batch_size, norm_all_data.shape[1]-1)
        self.x1min, self.x1max, self.x2min, self.x2max = np.min(norm_all_data[:,13]), np.max(norm_all_data[:,13]), np.min(norm_all_data[:,12]), np.max(norm_all_data[:,12])

        # TODO experiment with these
        optimizer = Adam(lr=0.001, beta_1=0.9)

        self.adversary = self.build_adversary()
        self.adversary.compile(loss=self.adversary_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        self.privatizer = self.build_privatizer()

        self.x = Input(shape=self.input_shape)
        self.y = self.privatizer(self.x)
        self.uhat = self.adversary(self.y)

    def adversary_loss(self, u, uhat):
        return keras.losses.categorical_crossentropy(u, uhat)

    def build_privatizer(self):
        sigma = (self.norm_clip/self.epsilon)*math.sqrt(2*math.log(1.25/self.delta))
        def priv(x):
            y = x + tf.random.normal([1, self.batch_size, 25], mean=0.0, stddev=sigma)
            return y
        return priv

    def build_adversary(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.input_shape[1]))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.num_users, activation='softmax'))

        # model.summary()

        y = Input(shape=self.input_shape)
        uhat = model(y)

        return Model(y, uhat)

    def train(self, adversary_epochs, seed=False):

        # load dataset
        X_train = self.norm_all_data[:,:25]
        print(X_train.shape)
        u = to_categorical(self.all_data[:,25])
        print(u.shape)

        for epoch in range(adversary_epochs):

            # Select a random batch
            if seed:
                np.random.seed(0)
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            X_train_batch = X_train[idx].reshape(1, self.batch_size, 25)
            u_train_batch = u[idx].reshape(1, self.batch_size, self.num_users)

            # perform l2 norm clipping
            X_train_batch = tf.clip_by_norm(X_train_batch,self.norm_clip,axes=2)
            X_train_batch = tf.cast(X_train_batch, tf.float32)

            # DP gaussian mechanism
            # Y_batch = self.privatizer.predict(X_train_batch, steps=1)
            Y_batch = self.privatizer(X_train_batch)

            # Train the adversary
            a_loss = self.adversary.train_on_batch(Y_batch, u_train_batch)

            # log the progress
            if epoch % 10 == 0:
                print ("%d [A loss: %f, acc.: %.2f%%]" % (epoch, a_loss[0], 100*a_loss[1]))


    def show_batch(self, seed=False):
        X = self.norm_all_data[:,:25]
        # Select a random batch
        if seed:
            np.random.seed(0)
        idx = np.random.randint(0, X.shape[0], self.batch_size)
        X_batch = X[idx].reshape(1, self.batch_size, 25)
        # perform l2 norm clipping
        X_batch = tf.clip_by_norm(X_batch,self.norm_clip,axes=2)
        X_batch = tf.cast(X_batch, tf.float32)
        # DP gaussian mechanism
        # Y_batch = self.privatizer.predict(X_batch, steps=1)
        Y_batch = self.privatizer(X_batch)

        X_batch = X_batch.eval(session=keras.backend.get_session())
        Y_batch = Y_batch.eval(session=keras.backend.get_session())

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].scatter(X_batch[0,:,13], X_batch[0,:,12], c=X_batch[0,:,6].tolist())
        ax[0].set_title("Input Data")
        ax[1].scatter(Y_batch[0,:,13], Y_batch[0,:,12], c=Y_batch[0,:,6].tolist())
        ax[1].set_title("Obfuscated Data")
        # plt.colorbar()
        plt.show()

    def tf_make_gridpoints(self):
            testpoints = np.mgrid[self.x1min:self.x1max:30j, self.x2min:self.x2max:30j].reshape(2,-1).T
            return tf.constant(testpoints, dtype=tf.float32)

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

    def batch_utility_loss(self, seed=False):
        X = self.norm_all_data[:,:25]
        # Select a random batch
        if seed:
            np.random.seed(0)
        idx = np.random.randint(0, X.shape[0], self.batch_size)
        X_batch = X[idx].reshape(1, self.batch_size, 25)
        # perform l2 norm clipping
        X_batch = tf.clip_by_norm(X_batch,self.norm_clip,axes=2)
        X_batch = tf.cast(X_batch, tf.float32)
        # DP gaussian mechanism
        # Y_batch = self.privatizer.predict(X_batch, steps=1)
        Y_batch = self.privatizer(X_batch)

        vx, bx, gridx, v_gridx, rss_predx = self.tf_fitmap(X_batch[0], 2)
        vy, by, gridy, v_gridy, rss_predy = self.tf_fitmap(Y_batch[0], 2)
        utility_loss = tf.reduce_mean(tf.square(by-bx))
        return utility_loss.eval(session=keras.backend.get_session())

    def average_utility_loss(self, batches=1000):
        t_loss = 0
        for i in range(batches):
            if i%10 == 0:
                print(i)
            t_loss += self.batch_utility_loss()
        return t_loss/batches

all_data = np.genfromtxt('augmented_data')
norm_all_data = np.genfromtxt('normalized_augmented_data')

print("setting up...")
# TODO choose norm_clip
n = NOISE_PRIVATIZER(all_data, norm_all_data, num_users=9, batch_size=512, epsilon=0.1, delta=10**-5, norm_clip=4.0)

#### NOTES ON PERFORMANCE ####
# batch_size=128, epsilon=0.1, delta=10**-5, norm_clip=4.0
# 0 [A loss: 2.811357, acc.: 10.94%]
# batch_size=512, epsilon=0.1, delta=10**-5, norm_clip=4.0
# 990 [A loss: 1.722018, acc.: 28.91%]

# batch_size=512, epsilon=0.1, delta=10**-5, norm_clip=4.0
# 990 [A loss: 1.705143, acc.: 31.64%]
# 119.26503776454925


# print("training adversary")
# n.train(adversary_epochs=1000, seed=False)

print("calculating utility of this privatizer")
print(n.average_utility_loss(1000))

# print("demonstrating privatizer on one batch")
# n.show_batch()
