import numpy as np
from matplotlib.dates import datestr2num
from signalmaps_helper import *

print("reading")
user0 = np.genfromtxt('Chania/1e33.csv', delimiter=',', skip_header=1, converters={0:lambda x: datestr2num(x.decode('UTF-8'))})
user1 = np.genfromtxt('Chania/2222.csv', delimiter=',', skip_header=1, converters={0:lambda x: datestr2num(x.decode('UTF-8'))})
user2 = np.genfromtxt('Chania/6882.csv', delimiter=',', skip_header=1, converters={0:lambda x: datestr2num(x.decode('UTF-8'))})
user3 = np.genfromtxt('Chania/7cbc.csv', delimiter=',', skip_header=1, converters={0:lambda x: datestr2num(x.decode('UTF-8'))})
user4 = np.genfromtxt('Chania/a841.csv', delimiter=',', skip_header=1, converters={0:lambda x: datestr2num(x.decode('UTF-8'))})

print("filtering")

user2 = user2[np.where(user2[:,12]>0)]
user2 = user2[np.where(user2[:,6]<0)]
user3 = user3[np.where(user3[:,12]>35.48)]
user0 = user0[np.where(user0[:,6]<0)]

user0 = np.concatenate((user0, np.zeros((len(user0), 1))), axis=1)
user1 = np.concatenate((user1, np.zeros((len(user1), 1))+1), axis=1)
user2 = np.concatenate((user2, np.zeros((len(user2), 1))+2), axis=1)
user3 = np.concatenate((user3, np.zeros((len(user3), 1))+3), axis=1)
user4 = np.concatenate((user4, np.zeros((len(user4), 1))+4), axis=1)

all_data = np.concatenate((user0,user1,user2,user3,user4), axis=0)

data_60562 = all_data[np.where(all_data[:,19]==60562)]

norm_all_data = norm_x(all_data, changedB=True)
norm_all_data_not_inverse = norm_x(all_data)
norm_data_60562 = norm_x(data_60562, changedB=True)
norm_data_60562_not_inverse = norm_x(data_60562)

ct = cell_tower(all_data)
ct_60562 = cell_tower(data_60562)

print("writing")

np.savetxt('alldata', all_data)
np.savetxt('norm_all_data_not_inverse', norm_all_data_not_inverse)
