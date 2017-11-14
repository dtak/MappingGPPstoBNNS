from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr

from blackbox_svi import vlb_inference
from autograd.misc.optimizers import adam
import plotting
from util import build_toy_dataset
from models import morph_bnn
from models import morph_bnn
import os
import seaborn as sns
sns.set_style('white')
rs = npr.RandomState(0)


if __name__ == '__main__':

    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)

    num_weights, predictions, logprob, _, _ = morph_bnn(layer_sizes=[1, 20, 20, 1],
                                                        nonlinearity=rbf)

    exp_num = 1
    data = 'cosx'  # or expx or cosx
    N_data = 70
    samples = 20
    iters = 100
    save_plot = True
    plot_during_training = False

    inputs, targets = build_toy_dataset(data, N_data)
    log_posterior = lambda weights, t: logprob(weights, inputs, targets)
    objective, gradient, unpack_params = vlb_inference(log_posterior, num_weights, samples)

    if plot_during_training:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)


    def callback(params, t, g):
        # Sample functions from posterior.
        mean, log_std = unpack_params(params)
        sample_weights = rs.randn(5, num_weights) * np.exp(log_std) + mean
        plot_inputs = np.linspace(-8, 8, num=400)
        outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))
        p = outputs[:, :, 0].T

        # Plot data and functions.
        if plot_during_training:
            plt.cla()
            ax.plot(inputs.ravel(), targets.ravel(), 'k.')
            ax.plot(plot_inputs, p, color='b')
            ax.set_ylim([-2, 3])
            plt.draw()
            plt.pause(1.0 / 60.0)

        print("Iteration {} | objective {}".format(t, objective(params, t)))

    # Initialize variational parameters
    rs = npr.RandomState(0)
    init_mean = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    var_params = adam(gradient, init_var_params,
                      step_size=0.1, num_iters=iters, callback=callback)

    if save_plot:
        N_data = 400
        N_samples = 5
        D = (inputs.ravel(), targets.ravel())
        x_plot = np.linspace(-8, 8, num=N_data)
        save_title = "exp-" + str(exp_num)+data
        save_dir = os.path.join(os.getcwd(), 'plots', 'bnn', save_title)

        # predictions from posterior of bnn
        mean, log_std = unpack_params(var_params)
        sample_weights = rs.randn(N_samples, num_weights) * np.exp(log_std) + mean
        p = predictions(sample_weights, x_plot[:, None])[:, :, 0].T
        plotting.plot_deciles(x_plot, p, D, save_dir, plot="bnn")
