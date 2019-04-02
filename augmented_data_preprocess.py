import numpy as np
from matplotlib.dates import datestr2num
from signalmaps_helper import *

userID = {
'a841f74e620f74ec443b7a25d7569545':0,
'22223276ea84bbce3a62073c164391fd':1,
'510635002cb29804d54bff664cab52be':2,
'7cbc37da05801d46e7d80c3b99fd5adb':3,
'7023889b4439d2c02977ba152d6f4c6e':4,
'8425a81da55ec16b7f9f80c139c235a2':5,
'6882f6cf8c72d6324ba7e6bb42c9c7c2':6,
'1e33db5d2be36268b944359fbdbdad21':7,
'892d2c3aae6e51f23bf8666c2314b52f':8,
}


cols = (0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25)
print("reading everything but the userIDs")
data = np.genfromtxt('augmented_data.csv', delimiter=',', skip_header=1, usecols = cols, converters={0:lambda x: datestr2num(x.decode('UTF-8'))})
print("reading userIDs")
userlabels = np.genfromtxt('augmented_data.csv', delimiter=',', skip_header=1, usecols = (1), converters={1:lambda x: userID[x.decode('UTF-8')]})

print(data[0:5])
print(userlabels[0:5])
print(data.shape)
print(userlabels.shape)
all_data = np.concatenate((data, np.transpose(np.array([userlabels]))), axis=1)
# filter data for usable entries
all_data = all_data[~np.isnan(all_data).any(axis=1)]

norm_all_data = norm_x(all_data)
ct = cell_tower(all_data)
print("The normalized cell tower for this dataset is at location:")
print(ct)
# [-1.52582084, -0.80501811]

print("writing")
np.savetxt('augmented_data', all_data)
np.savetxt('normalized_augmented_data', norm_all_data)
