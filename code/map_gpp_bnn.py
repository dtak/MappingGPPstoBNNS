import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky
from autograd.misc.optimizers import adam, sgd
from blackbox_svi import kl_inference, vlb_inference
from util import covariance, build_toy_dataset
import plotting
from plotting import plot_mean_std, plot_priors, plot_deciles, plot_samples
from models import morph_bnn
import os
import seaborn as sns
sns.set_style('white')
rs = npr.RandomState(0)

if __name__ == '__main__':
    rs = npr.RandomState(0)
    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)

    exp_num = 20
    data = 'xsinx'  # or expx or cosx
    N_data = 400
    samples = 20
    iters_1 = 20
    iters_2 = 50

    save_plot = True
    plot_during_training = True

    num_weights, predictions, _, log_p_gp, log_post = morph_bnn(layer_sizes=[1, 20, 20, 1],
                                                                nonlinearity=rbf)

    inputs, targets = build_toy_dataset(data, n_data=N_data)

    log_gp_prior = lambda w, t: log_p_gp(w, N_data)
    kl, grad_kl, unpack_params = kl_inference(log_gp_prior,
                                              N_weights=num_weights,
                                              N_samples=20)

    # set up fig
    if plot_during_training:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(params, t, g, title, objective):
        # Sample functions from prior or posterior f ~ p(f|phi) or p(f|varphi)
        N_samples = 3
        mean, log_std = unpack_params(params)
        sample_weights = rs.randn(N_samples, num_weights) * np.exp(log_std) + mean
        plot_inputs = np.linspace(-8, 8, num=400)
        outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))
        p = outputs[:, :, 0].T

        if "prior" in title:
            N_data = len(plot_inputs)
            K = covariance(plot_inputs[:, None], plot_inputs[:, None])
            L = cholesky(K + 1e-7 * np.eye(N_data))
            e = np.random.normal(size=(N_data, N_samples))
            f_gp_prior = np.dot(L, e)

        # Plot data and functions.
        if plot_during_training:
            plt.cla()
            ax.plot(inputs.ravel(), targets.ravel(), 'ko')
            ax.plot(plot_inputs, p, color='r')
            if "prior" in title:
                ax.plot(plot_inputs, f_gp_prior, color='g')
            ax.set_title(title)
            ax.set_ylim([-2, 3])
            plt.draw()
            plt.pause(1.0/40.0)

        print("Iteration {} | objective {}".format(t, objective(params, t)))

    # ----------------------------------------------------------------

    # Initialize the variational prior params HERE
    # the functions drawn from the optimized bnn prior are heavily
    # dependent on which initialization scheme used here

    init_mean = rs.randn(num_weights)
    init_log_std = -1.5 * np.ones(num_weights)
    #init_log_std = -1.5*rs.randn(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    # ---------------------- MINIMIZE THE KL --------------------------

    title = "Optimizing the prior"
    callback1 = lambda params, t, g: callback(params, t, g, title, kl)
    prior_params = adam(grad_kl, init_var_params,
                        step_size=0.1, num_iters=iters_1, callback=callback1)

    # --------------------- MINIMIZE THE VLB -----------------------------------

    print(np.round(prior_params-init_var_params, 3))
    title = "optimizing vlb"

    log_posterior = lambda weights, t: log_post(weights, inputs, targets, prior_params)
    vlb, grad_vlb, unpack_params = vlb_inference(log_posterior, N_weights=num_weights,
                                                                N_samples=samples)
    print(title)
    callback2 = lambda params, t, g: callback(params, t, g, title, vlb)

    init_mean = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    var_params = adam(grad_vlb, init_var_params,
                      step_size=0.1, num_iters=iters_2, callback=callback2)

    # PLOT STUFF BELOW HERE

    def get_prior_draws(x_plot, prior_param, samples=5):
        N_data = len(x_plot)

        # sample functions from the gp prior
        K = covariance(x_plot[:, None], x_plot[:, None])
        L = cholesky(K + 1e-7 * np.eye(N_data))
        f_gp_prior = np.dot(L, np.random.normal(size=(N_data, samples)))

        # sample functions from the bnn prior
        prior_weights = rs.randn(samples, num_weights)
        f_bnn_prior = predictions(prior_weights, x_plot[:, None])[:, :, 0].T

        # sample from the optimized bnn prior
        mean, log_std = unpack_params(prior_param)
        sample_weights = rs.randn(samples, num_weights) * np.exp(log_std) + mean
        f_gp_bnn_prior = predictions(sample_weights, x_plot[:, None])[:, :, 0].T

        draws = (f_gp_prior, f_bnn_prior, f_gp_bnn_prior)

        return draws

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
        mean, log_std = unpack_params(var_params)
        sample_weights = rs.randn(N_samples, num_weights) * np.exp(log_std) + mean
        p = predictions(sample_weights, x_plot[:, None])[:, :, 0].T

        save_dir = os.path.join(os.getcwd(), 'plots', 'gpp-bnn', save_title+data)
        # plot_samples(x_plot, p, D, save_dir, plot="gpp")
        # plot_mean_std(x_plot, p, D, save_dir, plot="gpp")
        plot_deciles(x_plot, p, D, save_dir, plot="gpp")


