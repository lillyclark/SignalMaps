## Differential Privacy

Notes on Performance  
```
Test 1
batch_size=512, epsilon=0.1, delta=10**-5, norm_clip=4.0
epochs: 990 [A loss: 1.722018, acc.: 28.91%]
Test 2
batch_size=512, epsilon=0.1, delta=10**-5, norm_clip=4.0
epochs: 990 [A loss: 1.705143, acc.: 31.64%]
Utility loss: 119.26503776454925
```

## Generative Adversarial Privacy

Notes on Performance  
```
Test 1  
rho=0.0, batch_size=128, epochs=11, adversary_epochs = 1
epochs: 10 [A loss: 2.544804, acc.: 7.81%] [P loss: -2.561621]
```

### Images Folder
1. The geographic spread of the data
```allpoint.png```
2. Sample output of the DP-privatizer
```epsilon0point1sample.png```
3. The effect of unincluding outlier RSS value datapoints when fitting a two degree polynomial to generate a signal map ```fitted2all.png, fitted2no_outliers.png```
4. Various polynomial fits (2 Degrees actually seems best)
```fitted{2no_outliers, 3, 4, 5, 6}.png```
5. The geographic spread of all the data after normalization with the cell tower location marked
```normpointscelltower.png```
6. The RSS values pre-filtering ```RSSallpoints.png```
7. Geographic spread of datapoints by user ```user{0:8}.png```
8. Histogram showing number of points per user ```userIDspread.png```
