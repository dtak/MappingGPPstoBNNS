
import theano
import theano.tensor as T
from black_box_alpha import BB_alpha

import os

import sys

import numpy as np

import gzip

import cPickle

# We download the data

def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'float32')

    #############
    # LOAD DATA #
    #############

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
    mean_y_train = np.mean(y_train, 0)
    std_y_train = np.std(y_train, 0)

    y_train = (y_train - mean_y_train) / std_y_train

    train_set = X_train, y_train
    test_set = X_test, y_test

    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval, train_set[ 0 ].shape[ 0 ], train_set[ 0 ].shape[ 1 ], mean_y_train, std_y_train

import os
if not os.path.isfile('results/test_error.txt') or not os.path.isfile('results/test_ll.txt'):

    os.system('rm results/*')

    # We load the random seed

    np.random.seed(1)

    # We load the data

    datasets, n, d, mean_y_train, std_y_train = load_data()

    train_set_x, train_set_y = datasets[ 0 ]
    test_set_x, test_set_y = datasets[ 1 ]

    N_train = train_set_x.get_value(borrow = True).shape[ 0 ]
    N_test = test_set_x.get_value(borrow = True).shape[ 0 ]
    layer_sizes = [ d, 50, 50, len(mean_y_train) ]
    n_samples = 50
    alpha = 0.5
    learning_rate = 0.002
    v_prior = 1.0
    batch_size = 250
    print '... building model'
    sys.stdout.flush()
    bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, \
        train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test, mean_y_train, std_y_train)
    print '... training'
    sys.stdout.flush()

    test_error, test_ll = bb_alpha.train_ADAM(1000)

    with open("results/test_ll.txt", "a") as myfile:
        myfile.write(repr(test_ll) + '\n')

    with open("results/test_error.txt", "a") as myfile:
        myfile.write(repr(test_error) + '\n')

    # We plot the training data and the predictions

    x = theano.function([], test_set_x)()
    y_predict = bb_alpha.sample_predictive_distribution(x)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axes = plt.gca()
    axes.set_ylim([ -15, 15 ])
    matplotlib.rcParams.update({'font.size': 15 })
    for i in range(y_predict.shape[ 0 ]):
        plt.plot(x, y_predict[ i, :, : ], '.', c = 'black', markersize = 0.75)
    plt.title('Predictions alpha = 0.5')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.savefig('results/figure.pdf', bbox_inches = 'tight')
