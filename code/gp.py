import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import grad
from autograd.misc.optimizers import  sgd, adam
import seaborn as sns
import os
import plotting
from kernels import kernel_dict as kernel
from util import build_toy_dataset
sns.set_style("white")

def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""
    rs = npr.RandomState(0)
    noise = 1e-4

    def unpack_kernel_params(params):
        mean        = params[0]
        cov_params  = params[2:]
        noise_scale = np.exp(params[1]) + noise
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar):
        """
        Returns the predictive mean and covariance at locations xstar,
        of the latent function value f (without observation noise).
        """
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x, xstar)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(params, x, y):
        """
        computes log p(y|X) = log N(y|mu, K + std*I)
        """
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)

    def sample_functions(params, x, y, xs, samples=5):
        pred_mean, pred_cov = predict(params, x, y, xs)
        marg_std = np.sqrt(np.diag(pred_cov))
        sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov, size=samples)
        return sampled_funcs.T

    return num_cov_params + 2, predict, log_marginal_likelihood, sample_functions

# Define an example covariance function.

def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x / lengthscales, 1) \
            - np.expand_dims(xp / lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))


if __name__ == '__main__':

    D = 1
    exp_num = 2
    n_data = 70
    iters = 5
    data = "expx"
    samples = 5
    save_plots = True
    plot_during = False
    rs = npr.RandomState(0)
    mvnorm = rs.multivariate_normal
    save_title = "exp-" + str(exp_num)+data + "-posterior samples {}".format(samples)
    save_dir = os.path.join(os.getcwd(), 'plots', 'gp', save_title)

    num_params, predict, log_marginal_likelihood, sample_f = \
        make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    X, y = build_toy_dataset(data, n_data)
    y = y.ravel()

    objective = lambda params, t: log_marginal_likelihood(params, X, y)

    if plot_during:
        fig = plt.figure(figsize=(12,8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.show(block=False)

    def callback(params, t, g):
        print("iteration {} Log likelihood {}".format(t, objective(params, t)))

        if plot_during:
            plt.cla()
            x_plot = np.reshape(np.linspace(-8, 8, 400), (400, 1))
            pred_mean, pred_cov = predict(params, X, y, x_plot)  # shapes [N_data], [N_data, N_data]
            std = np.sqrt(np.diag(pred_cov))  # shape [N_data]
            ax.plot(x_plot, pred_mean, 'b')
            ax.fill_between(x_plot.ravel(),
                            pred_mean - 1.96*std,
                            pred_mean + 1.96*std,
                            color=sns.xkcd_rgb["sky blue"])

            # Show sampled functions from posterior.
            sf = mvnorm(pred_mean, pred_cov, size=5)  # shape = [samples, N_data]
            ax.plot(x_plot, sf.T)

            ax.plot(X, y, 'k.')
            ax.set_ylim([-2, 3])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.draw()
            plt.pause(1.0/60.0)
            if t == 1:
                D = X, y[:, None]
                p = sample_f(params, X, y, x_plot, samples)
                plotting.plot_deciles(x_plot.ravel(), p, D, save_dir, plot="gp")


    # Initialize covariance parameters
    rs = npr.RandomState(0)
    init_params = 0.1 * rs.randn(num_params)
    cov_params = adam(grad(objective), init_params,
                      step_size=0.1, num_iters=iters, callback=callback)

    if save_plots:
        D = X, y[:, None]
        x_plot = np.reshape(np.linspace(-8, 8, 400), (400, 1))
        p = sample_f(cov_params, X, y, x_plot, samples)
        print(p.shape)
        plotting.plot_deciles(x_plot.ravel(), p, D, save_dir, plot="gp")







