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

class GAP():

    def __init__(self, all_data, norm_all_data, num_users, batch_size, rho):
        self.num_users = num_users
        self.rho = rho
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
        # x = Input(shape=(None, self.input_shape[1]))
        x = Input(self.input_shape)
        y = self.privatizer(x)
        self.adversary.trainable = False
        self.X_batch, self.Y_batch = x, y
        uhat = self.adversary(y)
        self.combined = Model(x, uhat)
        self.combined.compile(loss=self.privatizer_loss,optimizer=optimizer)

    def adversary_loss(self, u, uhat):
        return keras.losses.categorical_crossentropy(u, uhat)

    def privatizer_loss(self, u, uhat):
        X, Y = self.X_batch[0], self.Y_batch[0]
        v = vandermonde_tensor(X, 2)
        b = beta_tensor(v, X[:,6])
        v_obf = vandermonde_tensor(Y, 2)
        b_obf = beta_tensor(v_obf,Y[:,6])
        self.beta_loss = tf.reduce_mean(tf.square(b-b_obf))
        self.distortion_loss = tf.reduce_mean(tf.square(X-Y))
        self.geographic_distortion = tf.reduce_mean(tf.square(X[:,12:14]-Y[:,12:14]))
        privacy_loss = keras.losses.categorical_crossentropy(u, uhat)
        num_grids = 1
        count_diff = tf.Variable(0, dtype=tf.float32)
        size1 = self.x1max-self.x1min
        size2 = self.x2max-self.x2min
        for i in range(num_grids):
            for j in range(num_grids):
                a = self.x1min+(size1/num_grids*i)
                b = self.x1min+(size1/num_grids*(i+1))
                c = self.x2min+(size2/num_grids*j)
                d = self.x2min+(size2/num_grids*(j+1))
                X_count = tf.count_nonzero(tf.where((X[:,13] >= a ) & (X[:,13] < b) & (X[:,12] >= c) & (X[:,12] < d), X, tf.zeros_like(X)), dtype=tf.float32)
                Y_count = tf.count_nonzero(tf.where((Y[:,13] >= a ) & (Y[:,13] < b) & (Y[:,12] >= c) & (Y[:,12] < d), Y, tf.zeros_like(Y)), dtype=tf.float32)
                tf.assign_add(count_diff, tf.square(X_count-Y_count))
        self.geographic_density = count_diff/(num_grids*num_grids)
        w_map, w_dist, w_geodist, w_geodens = 1, 1, 1, 1
        return self.rho*(w_map*self.beta_loss + w_dist*self.distortion_loss + w_geodist*self.geographic_distortion + w_geodens*self.geographic_density) - (1-self.rho)*privacy_loss

    def build_privatizer(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.input_shape[1]))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.input_shape[1]))
        x = Input(shape=(None, self.input_shape[1]))
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
        model.add(Dense(self.num_users, activation='softmax'))
        y = Input(shape=(None, self.input_shape[1]))
        uhat = model(y)
        return Model(y, uhat)

    def split_data(self, train_portion):
        idx = np.arange(self.n)
        np.random.shuffle(idx)
        k = int(train_portion*self.n)
        trainidx = idx[:k]
        testidx = idx[k:]
        self.X = self.norm_all_data[:,:25]
        self.u = to_categorical(self.all_data[:,25])
        self.X_train = self.X[trainidx]
        self.X_test = self.X[testidx]
        self.Y_test = None # placeholder
        self.u_train = self.u[trainidx]
        self.u_test = self.u[testidx]
        extra_epochs = 0 #1000
        self.epochs = int(k/self.batch_size)+extra_epochs

    def train(self):
        print("training for", self.epochs, "epochs")
        self.init_flag = False
        for epoch in range(self.epochs): #TODO
            # Select a random batch
            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            self.X_batch = self.X_train[idx].reshape(1, self.batch_size, 25)
            # Generate Y
            self.Y_batch = self.privatizer.predict(self.X_batch)
            self.u_batch = self.u_train[idx].reshape(1, self.batch_size, self.num_users)
            # Train the adversary
            self.adversary.trainable = True
            for e in range(1): #TODO
                a_loss = self.adversary.train_on_batch(self.Y_batch, self.u_batch)
            self.adversary.trainable = False
            # Train the privatizer
            p_loss = self.combined.train_on_batch(self.X_batch, self.u_batch)
            # log the progress
            if epoch % 100 == 0: #TODO
                print ("%d [A loss: %f, acc.: %.2f%%] [P loss: %f]" % (epoch, a_loss[0], 100*a_loss[1], p_loss))
                # self.eval_privacy()
                # self.eval_utility()

    def eval_privacy(self):
        X = self.X_test
        if self.Y_test is None:
            self.Y_test = self.privatizer.predict(X.reshape(1, X.shape[0], X.shape[1]))
        Y = self.Y_test
        u = self.u_test
        a_loss = self.adversary.evaluate(Y, u.reshape(1, u.shape[0], u.shape[1]), verbose=0)
        print("PRIVACY LOSS:", a_loss[0])

    def eval_utility(self):
        X = self.X_test
        u = self.u_test
        p_loss = self.combined.evaluate(X.reshape(1, X.shape[0], X.shape[1]), u.reshape(1, u.shape[0], u.shape[1]), verbose=0)
        print("UTILITY LOSS 1 (error in map fitting):", self.beta_loss.eval(session=keras.backend.get_session()))
        print("UTILITY LOSS 2 (full dataset distance):", self.distortion_loss.eval(session=keras.backend.get_session()))
        print("UTILITY LOSS 3 (just geographic distance):", self.geographic_distortion.eval(session=keras.backend.get_session()))
        print("UTILITY LOSS 4 (geographic density):", self.geographic_density.eval(session=keras.backend.get_session()))

    def showplots(self):
        X = self.X_test
        if self.Y_test is None:
            self.Y_test = self.privatizer.predict(X.reshape(1, X.shape[0], X.shape[1]))
        Y = self.Y_test[0]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].scatter(X[:,13], X[:,12], c=X[:,6].tolist())
        ax[0].set_title("Input Data")
        ax[1].scatter(Y[:,13], Y[:,12], c=Y[:,6].tolist())
        ax[1].set_title("Obfuscated Data")
        plt.show()

all_data = np.genfromtxt('augmented_data')
norm_all_data = np.genfromtxt('normalized_augmented_data')

# rho is emphasis on utility
n = GAP(all_data, norm_all_data, num_users=9, batch_size=512, rho=1.0)
n.split_data(train_portion = 0.8)
n.train()
# n.eval_utility()
# n.eval_privacy()
# n.showplots()
