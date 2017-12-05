import matplotlib.pyplot as plt
import os
import numpy as np
from autograd.numpy.linalg import solve, cholesky
from util import covariance
import seaborn as sns
from scipy.stats import norm
sns.set_style("white")

n = 9
bnn_col = ["deep sky blue", "bright sky blue"]
gpp_bnn_col = ["red", "salmon"]
gp_col = ["green", "light green"]
colors = {"bnn": bnn_col, "gpp": gpp_bnn_col, "gp": gp_col}
sample_col = {"bnn": "bright sky blue", "gpp": "watermelon", "gp": "light lime"}
pal_col = {"bnn": sns.light_palette("#3498db", n_colors=n),  # nice blue
           "gpp": sns.light_palette("#e74c3c", n_colors=n),  # nice red
           "gp" : sns.light_palette("#2ecc71", n_colors=n)}  # nice green eh not so nice


def set_up(show=True):
    fig = plt.figure(figsize=(20, 8), facecolor='white')
    ax = fig.add_subplot(211, frameon=True)
    bx = fig.add_subplot(212, frameon=True)

    if show:
        plt.show(block=False)

    axes = (ax, bx)

    return fig, axes


def plot_mean_std(x_plot, p, D, title="", plot="bnn"):
    x, y = D
    col = colors[plot]

    mean = np.mean(p, axis=1)
    std = np.std(p, axis=1)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    # plot data
    ax.plot(x, y, 'ko', ms=4)

    # plot mean of samples
    ax.plot(x_plot, mean, sns.xkcd_rgb[col[0]], lw=2)

    # plot 95 confidence interval as shaded region
    ax.fill_between(x_plot, mean - 1.96 * std, mean + 1.96 * std,
                    color=sns.xkcd_rgb[col[1]])

    ax.tick_params(labelleft='off', labelbottom='off')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-8, 8])

    plt.savefig(title+plot+"-95 confidence.jpg", bbox_inches='tight')


def plot_deciles(x_plot, p, D, title="", plot="bnn"):
    x, y = D
    color = colors[plot]

    mean = np.mean(p, axis=1)
    std = np.std(p, axis=1)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    # Get critical values for the deciles
    lvls = 0.1*np.linspace(1, 9, 9)
    alphas = 1-0.5*lvls
    zs = norm.ppf(alphas)

    # plot data
    ax.plot(x, y, 'ko', ms=4)

    # plot samples
    ax.plot(x_plot, p, sns.xkcd_rgb[sample_col[plot]], lw=1)

    # plot mean of samples
    ax.plot(x_plot, mean, sns.xkcd_rgb[color[0]], lw=1)

    # plot the deciles
    pal = pal_col[plot]
    for z, col in zip(zs, pal):
        ax.fill_between(x_plot, mean - z * std, mean + z * std, color=col)
    ax.tick_params(labelleft='off', labelbottom='off')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-8, 8])

    plt.savefig(title+plot+"-deciles.pdf", bbox_inches='tight')


def plot_samples(x_plot, p, D, title="", plot="bnn"):
    x, y = D

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    # plot data
    ax.plot(x, y, 'ko', ms=4)

    # plot samples
    ax.plot(x_plot, p, sns.xkcd_rgb[sample_col[plot]], lw=1)

    ax.tick_params(labelleft='off', labelbottom='off')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-8, 8])

    plt.savefig(title+plot+"-samples.pdf", bbox_inches='tight')


def plot_priors(x_plot, draws, title):

    f_gp_prior, f_bnn_prior, f_gp_bnn_prior = draws
    f, ax = plt.subplots(3, sharex=True)

    # plot samples
    ax[0].plot(x_plot, f_gp_prior, sns.xkcd_rgb["green"], lw=1)
    ax[1].plot(x_plot, f_gp_bnn_prior, sns.xkcd_rgb["red"], lw=1)
    ax[2].plot(x_plot, f_bnn_prior, sns.xkcd_rgb["blue"], lw=1)


    # plot data distribution
    # ax[2].fill_between(x, -3, 3, facecolor=sns.xkcd_rgb["silver"])

    plt.tick_params(labelbottom='off')
    plt.xlim([-8, 8])
    plt.savefig(title+"samples.jpg", bbox_inches='tight')


def plot_gp_posterior(x, xtest, y, s=1e-4, samples=10, title="", plot='gp'):
    N = len(x); print(N)
    n = len(xtest)
    K = covariance(x, x) + s*np.eye(N)
    print(K.shape)
    L = cholesky(K)

    # compute the mean at our test points.
    Lk = solve(L, covariance(x, xtest))
    mu = np.dot(Lk.T, solve(L, y))

    # compute the variance at our test points.
    K_ = covariance(xtest, xtest)
    var = np.diag(K_) - np.sum(Lk ** 2, axis=0)
    std = np.sqrt(var)

    # draw samples from the prior at our test points.
    L = cholesky(K_ + s * np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, samples)))

    L = cholesky(K_ + s * np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu + np.dot(L, np.random.normal(size=(n, samples)))

    # --------------------------PLOTTING--------------------------------

    # PLOT PRIOR
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    ax.plot(x, y, 'ko', ms=4)

    # Get critical values for the deciles
    lvls = 0.1*np.linspace(1, 9, 9)
    alphas = 1-0.5*lvls
    zs = norm.ppf(alphas)
    pal = pal_col[plot]
    cols = colors[plot]

    print(f_prior.shape)
    print(f_post.shape)
    # plot samples, mean and deciles
    mean = np.mean(f_prior, axis=1)
    std = np.std(f_prior,axis=1)
    ax.plot(xtest, f_prior, sns.xkcd_rgb[sample_col[plot]], lw=1)

    ax.plot(xtest, mean, sns.xkcd_rgb[cols[0]], lw=1)
    print(xtest.shape, mean.shape, std.shape)
    for z, col in zip(zs, pal):
        ax.fill_between(xtest.ravel(), mean - z * std, mean + z * std, color=col)

    plt.tick_params(labelbottom='off')
    plt.xlim([-8, 8])
    plt.legend()
    plt.savefig(title+"GP prior_draws.pdf", bbox_inches='tight')

    # PLOT POSTERIOR
    plt.clf()
    std = np.sqrt(var)
    fig = plt.figure()
    bx = fig.add_subplot(111)

    bx.plot(x, y, 'ko', ms=4)
    print(col[0])
    # plot samples, mean and deciles
    bx.plot(xtest, f_post, sns.xkcd_rgb[sample_col[plot]], lw=1)
    # bx.plot(xtest, mu, sns.xkcd_rgb[cols[0]], lw=1)
    print(xtest.shape, mu.shape, std.shape)
    mu=mu.ravel()
    #for z, col in zip(zs, pal):
    #    bx.fill_between(xtest.ravel(), mu - z * std, mu + z * std, color=col)

    plt.tick_params(labelbottom='off')
    plt.xlim([-8, 8])
    plt.ylim([-2,3])
    plt.legend()
    plt.savefig(title + "GP post_draws.pdf", bbox_inches='tight')


