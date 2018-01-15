import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.numpy.linalg import solve, cholesky
from autograd.misc.optimizers import adam, sgd
from blackbox_svi import kl_inference, vlb_inference
from util import build_toy_dataset, make_title
import plotting
from models import map_gpp_bnn, construct_bnn
import os
rs = npr.RandomState(0)

if __name__ == '__main__':
    rs = npr.RandomState(0)

    # define nonlinearity here
    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)
    sigmoid = lambda x: 0.5*(np.tanh(x)**2-1)
    linear = lambda x: x
    softp = lambda x: np.log(1+np.exp(x))

    exp_num = 8
    x_num = 36
    samples = 20
    arch = [1, 20, 20, 1]
    act = "rbf"
    kern = "rbf"

    iters_1 = 40
    scale = -0.5
    step = 0.1

    save_plot = True
    save_during = False
    plot_during = True

    save_title = make_title(exp_num, x_num, samples, kern,
                            arch, act, iters_1, scale, step)

    save_dir = os.path.join(os.getcwd(), 'plots', 'exp', save_title)

    if save_during:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    num_weights, bnn_predict, unpack_params, \
    init_bnn_params, sample_bnn, sample_gpp, \
    kl, grad_kl = map_gpp_bnn(layer_sizes=arch, nonlinearity=rbf,
                              n_data=x_num, N_samples=samples, kernel=kern,
                              match_gradients=True)

    if plot_during:
        f, ax = plt.subplots(3, sharex=True)
        plt.ion()
        plt.show(block=False)

    def callback_kl(prior_params, iter, g):
        # Sample functions from priors f ~ p(f)
        n_samples = 3
        plot_inputs = np.linspace(-8, 8, num=500)

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

        fs = (f_gp, f_bnn, f_bnn_gpp)
        kl_val = kl(prior_params, iter)

        if save_during:
            title = " iter {} kl {:5}".format(iter, kl_val)
            plotting.plot_priors(plot_inputs, fs, os.path.join(save_dir, title))

        print("Iteration {} KL {} ".format(iter, kl_val))

    # ----------------------------------------------------------------
    # Initialize the variational prior params (phi) HERE for q(w|phi)

    init_var_params = init_bnn_params(num_weights, scale=scale)

    # ---------------------- MINIMIZE THE KL --------------------------

    prior_params = adam(grad_kl, init_var_params,
                        step_size=step, num_iters=iters_1, callback=callback_kl)


    # --------------------- MINIMIZE THE VLB -----------------------------------

    # Set up
    data = 'xsinx'  # or expx or cosx
    iters_2 = 100
    N_data = 70
    inputs, targets = build_toy_dataset(data, n_data=N_data)

    min_vlb = True


    if min_vlb:
        plot_during_ = True

        # construct the BNN
        N_weights, init_bnn_params, predictions, sample_bnn, \
        log_post, unpack_params, vlb_objective = construct_bnn(layer_sizes=arch, nonlinearity=rbf)

        log_posterior = lambda weights, t: log_post(weights, inputs, targets, prior_params)
        vlb = lambda param, t: vlb_objective(param, log_posterior, t)


        def save(params, t):
            D = (inputs.ravel(), targets.ravel())
            x_plot = np.linspace(-8, 8, num=400)
            save_title = "exp-" + str(exp_num) + "iter " + str(t)

            # predictions from posterior of bnn
            p = sample_bnn(x_plot, 5, params)

            save_dir = os.path.join(os.getcwd(), 'plots', 'gpp-bnn', save_title + data)
            plotting.plot_deciles(x_plot, p, D, save_dir, plot="gpp")


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

            if t>50:
                save(params, t)
            print("Iteration {} | vlb {}".format(t, -objective(params, t)))




        callback_vlb = lambda params, t, g: callback(params, t, g, vlb)

        init_var_params = init_bnn_params(num_weights)

        var_params = adam(grad(vlb), init_var_params,
                          step_size=0.1, num_iters=iters_2, callback=callback_vlb)

    # PLOT STUFF BELOW HERE


    if save_plot:
        N_data = 400
        N_samples = 5
        D = (inputs.ravel(), targets.ravel())
        x_plot = np.linspace(-8, 8, num=N_data)
        save_title = "exp-" + str(exp_num)


        # predictions from posterior of bnn
        p = sample_bnn(x_plot, N_samples, var_params)

        save_dir = os.path.join(os.getcwd(), 'plots', 'gpp-bnn', save_title+data)
        plotting.plot_deciles(x_plot, p, D, save_dir, plot="gpp")


