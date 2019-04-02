import numpy as np
import matplotlib.pyplot as plt
from signalmaps_helper import *

# all_data = np.genfromtxt('augmented_data')
# norm_all_data = np.genfromtxt('normalized_augmented_data')

## How many datapoints are there from each user?
# p = plt.hist(all_data[:,25], [0,1,2,3,4,5,6,7,8,9])
# plt.title("Datapoints per userID")
# plt.show()

## What is the geographic spread?
# p = plt.scatter(all_data[:,13], all_data[:,12])
# plt.title("All datapoints in geographic space")
# plt.show()

## Where is the cell tower in relation to all points?
# p = plt.scatter(norm_all_data[:,13], norm_all_data[:,12])
# p = plt.scatter([-1.52582084], [-0.80501811], marker='x')
# plt.title("All datapoints, cell tower marked")
# plt.show()

## Where is each user?
# for user in range(9):
#     u = np.where(all_data[:,25]==user)
#     p = plt.scatter(norm_all_data[u][:,13], norm_all_data[u][:,12])
#     plt.title("User %d" %user)
#     plt.show()

## Are there outliers in the RSS data?
# p = plt.plot(all_data[:,6])
# plt.title("Received Signal Strengths")
# plt.show()

## What does a map fitted to this look like?
# fitmap(norm_all_data[np.where(all_data[:,6]<-25)], 2, show=True, cell_tower=[-1.52582084, -0.80501811])
# fitmap(norm_all_data[np.where(all_data[:,6]<-25)], 3, show=True, cell_tower=[-1.52582084, -0.80501811])
# fitmap(norm_all_data[np.where(all_data[:,6]<-25)], 4, show=True, cell_tower=[-1.52582084, -0.80501811])
# fitmap(norm_all_data[np.where(all_data[:,6]<-25)], 5, show=True, cell_tower=[-1.52582084, -0.80501811])
# fitmap(norm_all_data[np.where(all_data[:,6]<-25)], 6, show=True, cell_tower=[-1.52582084, -0.80501811])
