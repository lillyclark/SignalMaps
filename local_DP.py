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
        self.n = norm_all_data.shape[0]
        self.input_shape = (batch_size, norm_all_data.shape[1]-1)
        self.x1min, self.x1max, self.x2min, self.x2max = np.min(norm_all_data[:,13]), np.max(norm_all_data[:,13]), np.min(norm_all_data[:,12]), np.max(norm_all_data[:,12])
        # TODO experiment with these
        optimizer = Adam(lr=0.001, beta_1=0.9)
        self.adversary = self.build_adversary()
        self.adversary.compile(loss=self.adversary_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.privatizer = self.build_privatizer()

    def adversary_loss(self, u, uhat):
        return keras.losses.categorical_crossentropy(u, uhat)

    def build_privatizer(self):
        # if unconcerned with DP guarantees, set sigma directly
        sigma = (self.norm_clip/self.epsilon)*math.sqrt(2*math.log(1.25/self.delta))
        def priv(x):
            y = x + tf.random.normal([self.n, 25], mean=0.0, stddev=sigma)
            # y = x + tf.random.normal([1, self.n, 25], mean=0.0, stddev=sigma)
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
        y = Input(shape=(None, self.input_shape[1]))
        uhat = model(y)
        return Model(y, uhat)

    def privatize(self):
        X = self.norm_all_data[:,:25]
        # X = X.reshape(1, self.n, 25)
        # X = tf.clip_by_norm(X, self.norm_clip, axes=2)
        X = tf.clip_by_norm(X, self.norm_clip, axes=1)
        self.X = tf.cast(X, tf.float32)
        self.Y = self.privatizer(self.X)
        self.u = to_categorical(self.all_data[:,25])

    def eval_utility(self):
        pass

    def split_data(self, train_portion):
        idx = np.random.randint(0, self.n, self.n)
        k = int(train_portion*self.n)
        trainidx = idx[:k]
        testidx = idx[k:]
        self.Y_train = self.Y[trainidx]
        self.Y_test = self.Y[testidx]
        self.u_train = self.u[trainidx]
        self.u_test = self.u[testidx]
        extra_epochs = 0
        self.adversary_epochs = int(k/self.batch_size)+extra_epochs

    def train(self, seed=False):
        for epoch in range(self.adversary_epochs):
            # Select a random batch
            if seed:
                np.random.seed(0)
            # TO DO slice tensor by random indices
            Y_batch = self.Y_train[:,self.batch_size*epoch:self.batch_size*(epoch+1),:]
            u_batch = self.u_train[:,self.batch_size*epoch:self.batch_size*(epoch+1),:]
            print(u_batch.shape)
            # Train the adversary
            a_loss = self.adversary.train_on_batch(Y_batch, u_batch)
            # log the progress
            if epoch % 100 == 0:
                print ("%d [A loss: %f, acc.: %.2f%%]" % (epoch, a_loss[0], 100*a_loss[1]))
                return

    def eval_privacy(self):
        Y = self.Y_test
        u = self.u_test
        a_loss = self.adversary.test_on_batch(Y.reshape(1, Y.shape[0], Y.shape[1]), u.reshape(1, u.shape[0], u.shape[1]))
        print("PRIVACY LOSS:", a_loss[0])

    def showplots(self):
        X = self.X.eval(session=keras.backend.get_session())
        Y = self.Y.eval(session=keras.backend.get_session())
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].scatter(X[0,:,13], X[0,:,12], c=X[0,:,6].tolist())
        ax[0].set_title("Input Data")
        ax[1].scatter(Y[0,:,13], Y[0,:,12], c=Y[0,:,6].tolist())
        ax[1].set_title("Obfuscated Data")
        # plt.colorbar()
        plt.show()

all_data = np.genfromtxt('augmented_data')
norm_all_data = np.genfromtxt('normalized_augmented_data')

n = NOISE_PRIVATIZER(all_data, norm_all_data, num_users=9, batch_size=512, epsilon=0.1, delta=10**-5, norm_clip=4.0)
n.privatize()
n.eval_utility()
n.split_data(train_portion = 0.8)
# n.train()
# n.eval_privacy()
# n.showplots()
