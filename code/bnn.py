from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from blackbox_svi import vlb_inference
from autograd.misc.optimizers import adam
import plotting
from util import build_toy_dataset
from models import construct_bnn
import os
import seaborn as sns
sns.set_style('white')
rs = npr.RandomState(0)


if __name__ == '__main__':

    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)


    # Set up
    exp_num = 1
    samples = 20
    arch = [1, 20, 20, 1]


    data = 'xsinx'  # or expx or cosx
    iters_2 = 50
    N_data = 70
    inputs, targets = build_toy_dataset(data, n_data=N_data)

    save_plot =True
    plot_during_ = True
    save_dir = os.path.join(os.getcwd(), 'plots', 'bnn', "exp-" + str(exp_num) + data)

    # construct the BNN
    N_weights, init_bnn_params, predictions, sample_bnn, \
    log_post, unpack_params, vlb_objective = construct_bnn(layer_sizes=arch, nonlinearity=rbf)

    log_posterior = lambda weights, t: log_post(weights, inputs, targets)
    vlb = lambda param, t: vlb_objective(param, log_posterior, t)

    # set up fig
    if plot_during_:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.ion()
        plt.show(block=False)


    def callback(params, t, g, objective):
        # Sample functions from posterior f ~ p(f|phi) or p(f|varphi)
        N_samples = 5
        plot_inputs = np.linspace(-8, 8, num=400)
        f_bnn = sample_bnn(plot_inputs, N_samples, params)

        # Plot data and functions.
        if plot_during_:
            plt.cla()
            ax.plot(inputs.ravel(), targets.ravel(), 'k.')
            ax.plot(plot_inputs, f_bnn, color='r')
            ax.set_title("fitting to toy data")
            ax.set_ylim([-5, 5])
            plt.draw()
            plt.pause(1.0 / 60.0)

        if t>25:
            D = (inputs.ravel(), targets.ravel())
            plotting.plot_deciles(plot_inputs, f_bnn, D, save_dir+"iter {}".format(t), plot="bnn")

        print("Iteration {} | vlb {}".format(t, -objective(params, t)))




    callback_vlb = lambda params, t, g: callback(params, t, g, vlb)

    init_var_params = init_bnn_params(N_weights)

    var_params = adam(grad(vlb), init_var_params,
                      step_size=0.1, num_iters=iters_2, callback=callback_vlb)




