import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky
from autograd.misc.optimizers import adam, sgd
from blackbox_svi import kl_inference, vlb_inference
from util import build_toy_dataset
import plotting
from plotting import plot_mean_std, plot_priors, plot_deciles, plot_samples
from models import map_gpp_bnn, construct_bnn
import os
import kernels
import seaborn as sns
sns.set_style('white')
rs = npr.RandomState(0)

if __name__ == '__main__':
    rs = npr.RandomState(0)

    # define nonlinearity here
    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)
    sigmoid = lambda x: 0.5*(np.tanh(x)**2-1)
    linear = lambda x: x

    exp_num = 26
    data = 'xsinx'  # or expx or cosx
    N_data = 70
    samples = 20
    iters_1 = 500
    iters_2 = 100

    save_plot = False
    plot_during = True

    num_weights, bnn_predict, unpack_params, \
    init_bnn_params, sample_bnn, sample_gpp, \
    kl, grad_kl = map_gpp_bnn(layer_sizes=[1, 30, 1], nonlinearity=np.sin)

    inputs, targets = build_toy_dataset(data, n_data=N_data)

    if plot_during:
        f, ax = plt.subplots(3, sharex=True)
        plt.ion()
        plt.show(block=False)

    def callback_kl(prior_params, iter, g):
        # Sample functions from priors f ~ p(f)
        n_samples = 3
        plot_inputs = np.linspace(-8, 8, num=100)

        f_bnn_gpp = sample_bnn(plot_inputs, n_samples, prior_params)    # f ~ p_bnn (f|phi)
        f_bnn     = sample_bnn(plot_inputs, n_samples)                  # f ~ p_bnn (f)
        f_gp      = sample_gpp(plot_inputs, n_samples)                  # f ~ p_gp  (f)

        # Plot samples of functions from the bnn and gp priors.
        if plot_during:
            for axes in ax: axes.cla()  # clear plots
            # ax.plot(x.ravel(), y.ravel(), 'ko')
            ax[0].plot(plot_inputs, f_gp, color='green')
            ax[1].plot(plot_inputs, f_bnn_gpp, color='red')
            ax[2].plot(plot_inputs, f_bnn, color='blue')
            #ax[0].set_ylim([-5, 5])
            #ax[1].set_ylim([-5, 5])
            #ax[2].set_ylim([-5, 5])

            plt.draw()
            plt.pause(1.0/40.0)

        print("Iteration {} KL {} ".format(iter, kl(prior_params, iter)))

    # ----------------------------------------------------------------

    # Initialize the variational prior params (phi) HERE for q(w|phi)
    # the functions drawn from the optimized bnn prior are heavily
    # dependent on which initialization scheme used here

    init_var_params = init_bnn_params(num_weights, scale=-0.5)


    # ---------------------- MINIMIZE THE KL --------------------------


    prior_params = adam(grad_kl, init_var_params,
                        step_size=0.15, num_iters=iters_1, callback=callback_kl)


    # --------------------- MINIMIZE THE VLB -----------------------------------


    min_vlb = False

    if min_vlb:

        # set up fig
        if plot_during:
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
            if plot_during:
                plt.cla()
                ax.plot(inputs.ravel(), targets.ravel(), 'k.')
                ax.plot(plot_inputs, f_bnn, color='r')
                ax.set_title("fitting to toy data")
                ax.set_ylim([-5, 5])
                plt.draw()
                plt.pause(1.0 / 60.0)

            print("Iteration {} | vlb {}".format(t, -objective(params, t)))

        log_posterior = lambda weights, t: log_post(weights, inputs, targets, prior_params)

        vlb, grad_vlb, unpack_params = \
            vlb_inference(log_posterior, N_weights=num_weights, N_samples=samples)

        callback_vlb = lambda params, t, g: callback(params, t, g, vlb)

        init_var_params = init_bnn_params(num_weights)

        var_params = adam(grad_vlb, init_var_params,
                          step_size=0.1, num_iters=iters_2, callback=callback_vlb)

    # PLOT STUFF BELOW HERE

    def get_prior_draws(x_plot, prior_param, samples=5):
        # sample functions from the gp, bnn, optimized bnn prior
        f_bnn_gpp = sample_bnn(x_plot, samples, prior_param)    # f ~ p_bnn (f|phi)
        f_bnn     = sample_bnn(x_plot, samples)                  # f ~ p_bnn (f)
        f_gp      = sample_gp_prior(x_plot, samples)             # f ~ p_gp  (f)
        return f_gp, f_bnn, f_bnn_gpp

    if save_plot:
        N_data = 400
        N_samples = 5
        D = (inputs.ravel(), targets.ravel())
        x_plot = np.linspace(-8, 8, num=N_data)
        save_title = "exp-" + str(exp_num)
        save_dir = os.path.join(os.getcwd(), 'plots', save_title)

        # SAVE 3x1 PLOT of function drawn from the 3 priors
        prior_p = get_prior_draws(x_plot, prior_params)
        plot_priors(x_plot, prior_p, D, save_dir)

        # predictions from posterior of bnn
        p = sample_bnn(x_plot, N_samples, var_params)

        save_dir = os.path.join(os.getcwd(), 'plots', 'gpp-bnn', save_title+data)
        plot_deciles(x_plot, p, D, save_dir, plot="gpp")


