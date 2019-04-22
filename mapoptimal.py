from signalmaps_helper import *
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical

# PARAMETERS
scramble_all_features = True
showplots = False
BATCH_SIZE = 512
train_portion = 0.7
extra_epochs = 3000

# read in dataset
all_data = np.genfromtxt('augmented_data')
norm_all_data = np.genfromtxt('normalized_augmented_data')

n = norm_all_data.shape[0]

# fit map to dataset
v = vandermonde(norm_all_data, 2)
b = beta(v, norm_all_data[:,6])

# generate location points iid in grid space
x1min, x1max, x2min, x2max = np.min(norm_all_data[:,13]), np.max(norm_all_data[:,13]), np.min(norm_all_data[:,12]), np.max(norm_all_data[:,12])
newx1 = np.random.uniform(x1min,x1max,n).reshape(1,n).T
newx2 = np.random.uniform(x2min,x2max,n).reshape(1,n).T
newpoints = np.concatenate((newx1, newx2),1)

# sample generated map distribution
v_newpoints = vandermonde(newpoints, 2)
newrss = np.dot(v_newpoints, b)

# populate and return new dataset
Y = norm_all_data.copy()
Y[:,13] = newx1[:,0]
Y[:,12] = newx2[:,0]
Y[:,6] = newrss[:,0]

if scramble_all_features:
    for feature in range(25):
        if feature == 12 or feature == 13 or feature == 6:
            pass
        else:
            start, end = np.min(norm_all_data[:,feature]), np.min(norm_all_data[:,feature])
            newfeature = np.random.uniform(start, end, n).reshape(1,n).T
            Y[:,feature] = newfeature[:,0]

if showplots:
    p = plt.scatter(norm_all_data[:,13], norm_all_data[:,12], c=norm_all_data[:,6])
    plt.colorbar()
    plt.title("Input Data (colored by RSS)")
    plt.show()

    p = plt.scatter(Y[:,13], Y[:,12], c=Y[:,6])
    plt.colorbar()
    plt.title("Obfuscated Data (colored by RSS)")
    plt.show()

# evaluate utility
# metric #1
v_eval = vandermonde(Y, 2)
b_eval = beta(v_eval,Y[:,6])
beta_loss = np.mean(np.square(b-b_eval))
print("UTILITY LOSS 1 (error in map fitting):", beta_loss)

distortion_loss = np.mean(np.square(norm_all_data-Y))
print("UTILITY LOSS 2 (full dataset distance):", distortion_loss)

geographic_distortion = np.mean(np.square(norm_all_data[:,12:14]-Y[:,12:14]))
print("UTILITY LOSS 3 (just geographic distance):", geographic_distortion)

# evaluate privacy
num_users = 9
input_shape = (BATCH_SIZE, 25)
model = Sequential()
model.add(Dense(32, input_dim=input_shape[1]))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(num_users, activation='softmax'))
y = Input(shape=(None, input_shape[1]))
uhat = model(y)

adversary = Model(y, uhat)
optimizer = Adam(lr=0.001, beta_1=0.9)
adversary.compile(loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])
u = to_categorical(all_data[:,25])

# controlling portion adversary can train on, to model side information
idx = np.arange(Y.shape[0])
np.random.shuffle(idx)
Y_train = Y[idx][0:int(train_portion*Y.shape[0])]
u_train = u[idx][0:int(train_portion*Y.shape[0])]
Y_test = Y[idx][int(train_portion*Y.shape[0]):]
u_test = u[idx][int(train_portion*Y.shape[0]):]

adversary_epochs = int((n*train_portion)/input_shape[0])

print("training for", adversary_epochs+extra_epochs, "epochs")
for epoch in range(adversary_epochs+extra_epochs):
    # Select a random batch
    idx = np.random.randint(0, Y_train.shape[0], input_shape[0])
    Y_batch = Y_train[:,:25][idx].reshape(1, input_shape[0], input_shape[1])
    u_batch = u_train[idx].reshape(1, input_shape[0], num_users)
    # Train the adversary
    a_loss = adversary.train_on_batch(Y_batch, u_batch)
    # log the progress
    if epoch % 100 == 0:
        print ("%d [A loss: %f, acc.: %.2f%%]" % (epoch, a_loss[0], 100*a_loss[1]))

# test adversary
a_loss = adversary.test_on_batch(Y_test[:,:25].reshape(1, Y_test.shape[0], input_shape[1]), u_test.reshape(1, u_test.shape[0], num_users))
print("PRIVACY LOSS:", a_loss[0])
