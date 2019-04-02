import numpy as np
import random
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

from math import ceil, log

def norm_x(x, changedB=False):
    norm_x = np.zeros_like(x)
    if changedB:
        print("mean of RSS: ", np.mean(x[:,6]), ", std of RSS: ", np.std(x[:,6]))
        rssinv = 1/(10**(x[:,6]/10))
        if np.std(rssinv) != 0:
            print("mean of RSS inv (W): ", np.mean(rssinv), ", std of RSS inv (W): ", np.std(rssinv))
            norm_x[:,6] = (rssinv-np.mean(rssinv))/np.std(rssinv)
        else:
            norm_x[:,6] = rssinv
    for col in range(x.shape[1]):
        if not (col == 6 and changedB):
            vals = x[:,col]
            if np.std(vals) != 0:
                norm_x[:,col] = (vals-np.mean(vals))/np.std(vals)
            else:
                norm_x[:,col] = vals
    return norm_x

def cell_tower(all_data):
    return np.array([(24.021793-np.mean(all_data[:,13]))/np.std(all_data[:,13]),
              (35.513226-np.mean(all_data[:,12]))/np.std(all_data[:,12])])

#### NP

def make_gridpoints(x):
        x1min, x1max = np.min(x[:,13]), np.max(x[:,13])
        x2min, x2max = np.min(x[:,12]), np.max(x[:,12])
        testpoints = np.mgrid[x1min:x1max:30j, x2min:x2max:30j].reshape(2,-1).T
        return testpoints

def vandermonde(X, num_features):
    poly = PolynomialFeatures(num_features)
    if X.shape[1] > 2:
        justcoords = np.array([X[:,13], X[:,12]]).T
        return poly.fit_transform(justcoords)
    return poly.fit_transform(X)

def beta(vandermonde_x, rssinv):
    beta = np.dot(np.linalg.inv(np.dot(np.transpose(vandermonde_x), vandermonde_x)),
                          np.dot(np.transpose(vandermonde_x), np.reshape(rssinv,[rssinv.shape[0],1])))
    return beta

def fitmap(data, num_degrees, show=False, cell_tower=None):
    v = vandermonde(data, num_degrees)
    b = beta(v, data[:,6])
    grid = make_gridpoints(data)
    v_grid = vandermonde(grid, num_degrees)
    rssinv_pred = np.dot(v_grid, b)
    if show:

        print("Predicted values are in the range")
        print(np.min(rssinv_pred), np.max(rssinv_pred))
        print("Actual values are in the range")
        print(np.min(data[:,6]), np.max(data[:,6]))

        fig = plt.figure()
        plt.scatter(np.concatenate([grid[:,0], data[:,13]]),
                    np.concatenate([grid[:,1], data[:,12]]),
                    c=np.concatenate([rssinv_pred[:,0], data[:,6]]).tolist())
        plt.colorbar()
        if cell_tower:
            plt.scatter(cell_tower[0], cell_tower[1], marker="X", c='red')
        plt.title("Map Fitted using %d Polynomial" %num_degrees)
        plt.show()
    return v, b, grid, v_grid, rssinv_pred
