from signalmaps_helper import *
import numpy as np
import matplotlib.pyplot as plt

# read in dataset
all_data = np.genfromtxt('augmented_data')
norm_all_data = np.genfromtxt('normalized_augmented_data')

n = norm_all_data.shape[0]

# fit map to dataset
# should I fit the pre-normalized data?
v = vandermonde(norm_all_data, 2)
b = beta(v, norm_all_data[:,6])

# generate location points iid in grid space
x1min, x1max, x2min, x2max = np.min(norm_all_data[:,13]), np.max(norm_all_data[:,13]), np.min(norm_all_data[:,12]), np.max(norm_all_data[:,12])
newx1 = np.random.uniform(x1min,x1max,n).reshape(1,n).T
newx2 = np.random.uniform(x2min,x2max,n).reshape(1,n).T
newpoints = np.concatenate((newx1, newx2),1)
print(newpoints.shape)

# sample generated map distribution
v_newpoints = vandermonde(newpoints, 2)
newrss = np.dot(v_newpoints, b)
print(newrss.shape)

# populate and return new dataset
Y = norm_all_data.copy()
Y[:,13] = newx1[:,0]
Y[:,12] = newx2[:,0]
Y[:,6] = newrss[:,0]

p = plt.scatter(norm_all_data[:,13], norm_all_data[:,12], c=norm_all_data[:,6])
plt.colorbar()
plt.title("Input Data (colored by RSS)")
plt.show()

p = plt.scatter(Y[:,13], Y[:,12], c=Y[:,6])
plt.colorbar()
plt.title("Obfuscated Data (colored by RSS)")
plt.show()
