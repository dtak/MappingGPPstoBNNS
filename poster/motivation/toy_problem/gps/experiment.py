import numpy as np
import scipy.stats    as sps

# We fix the random seed

np.random.seed(1)

# We load the data

X_train = np.loadtxt('../data/x_train.txt')[ : , None ]
y_train = np.loadtxt('../data/y_train.txt')[ : , None ]
X_test = np.loadtxt('../data/x_test.txt')[ : , None ]
y_test = np.loadtxt('../data/y_test.txt')[ : , None ]

# We normalize the features

std_X_train = np.std(X_train, 0)
std_X_train[ std_X_train == 0 ] = 1
mean_X_train = np.mean(X_train, 0)
X_train = (X_train - mean_X_train) / std_X_train
X_test = (X_test - mean_X_train) / std_X_train
X_train = np.concatenate((X_train, np.ones((X_train.shape[ 0 ], 1))), 1)
X_test = np.concatenate((X_test, np.ones((X_test.shape[ 0 ], 1))), 1)
mean_y_train = np.mean(y_train, 0)
std_y_train = np.std(y_train, 0)
std_y_train[ std_y_train == 0 ] = 1
y_train = (y_train - mean_y_train) / std_y_train
y_test_scaled = (y_test - mean_y_train) / std_y_train

y_train = np.array(y_train, ndmin = 2).reshape((-1, 1))
y_test = np.array(y_test, ndmin = 2).reshape((-1, 1))
y_test_scaled = np.array(y_test_scaled, ndmin = 2).reshape((-1, 1))

from dgps.layers.input_layer import *
from dgps.layers.output_layer_classification import *
from dgps.layers.output_layer_regression import *
from dgps.layers.output_layer_multiclass import *
from dgps.layers.noisy_layer import *
from dgps.layers.gp_layer import *
from dgps.gp_network import *

net = GP_Network(X_train, y_train[ : , : 1 ])
net.addInputLayer()
net.addGPLayer(150, 1)
net.addNoisyLayer()
net.addOutputLayerRegression()

net.train_via_ADAM(X_train, y_train[ : , : 1 ], X_test, y_test_scaled[ :, : 1 ],
    max_iterations = 2000, minibatch_size = 250, learning_rate = 0.001)

pred, uncert = net.predict(X_test)

error = np.sqrt(np.mean((pred * std_y_train[ 0 ] + mean_y_train[ 0 ] - y_test[ : , : 1 ])**2))
testll = np.mean(sps.norm.logpdf(pred * std_y_train[ 0 ] + mean_y_train[ 0 ] - y_test[ : , : 1 ], scale = np.sqrt(uncert) * std_y_train[ 0 ]))

# We load the hypers

with open("results/test_error.txt", "a") as myfile:
	myfile.write(repr(error) + '\n')

with open("results/test_ll.txt", "a") as myfile:
	myfile.write(repr(testll) + '\n')

# We plot the training data and the predictions

x = X_test
y_predict = np.expand_dims(pred * std_y_train[ 0 ] + mean_y_train[ 0 ], 0) + \
    np.expand_dims(np.sqrt(uncert) * std_y_train[ 0 ], 0) * np.random.randn(50, x.shape[ 0 ], 1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
axes = plt.gca()
axes.set_ylim([ -15, 15 ])
matplotlib.rcParams.update({'font.size': 15 })
for i in range(50):
    sample = pred * std_y_train[ 0 ] + mean_y_train[ 0 ] + np.sqrt(uncert) * std_y_train[ 0 ] * np.random.randn(x.shape[ 0 ], 1)
    plt.plot(x[ :, : 1 ], sample, '.', c = 'black', markersize = 0.75)

plt.title('Predictions Gaussian Process')
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('results/figure.pdf', bbox_inches = 'tight')
