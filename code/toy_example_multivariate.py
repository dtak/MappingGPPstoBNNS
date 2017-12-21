import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.misc.optimizers import adam
from autograd import grad
from autograd.scipy.stats import multivariate_normal as mvn

rs = npr.RandomState(0)

def unpack_layers(weights, layer_sizes):
    """ iterable that unpacks the weights into relevant tensor shapes for each layer"""
    shapes, _ = shapes_and_num(layer_sizes)
    num_weight_sets = len(weights)
    for m, n in shapes:
        yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
              weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
        weights = weights[:, (m+1)*n:]

def predictions(weights, inputs, layer_sizes, nonlinearity=np.tanh):
    inputs = np.expand_dims(inputs, 0)
    for W, b in unpack_layers(weights, layer_sizes):
        outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs

def empirical_cov(x):
    # x must be samples x dimensions
    centred = x - np.mean(x, axis=0, keepdims=True)
    return np.dot(centred.T, centred) / x.shape[0]

def entropy_estimate(samples):
    return mvn.entropy(np.zeros(samples.shape[1]), empirical_cov(samples))

def shapes_and_num(layer_sizes):
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m + 1) * n for m, n in shapes)
    return shapes, N_weights

def sample_normal(mean, log_std, N_samples):
    return rs.randn(N_samples, mean.shape[0]) * np.exp(log_std) + mean

def sample_obs(params, N_samples, x, layer_sizes, with_noist=True):
    mean, log_std = params
    bnn_weights = sample_normal(mean, log_std, N_samples)
    f_bnn = predictions(bnn_weights, x, layer_sizes)[:, :, 0]
    if with_noist:
        return f_bnn + noise_var * rs.randn(N_samples, x.shape[0])
    else:
        return f_bnn

def kl_estimate(params, N_samples, x, layer_sizes, mean, cov):
    y = sample_obs(params, N_samples, x, layer_sizes)
    return -entropy_estimate(y) - np.mean(mvn.logpdf(y, mean, cov))

def plot_isocontours(ax, func, xlimits=[-5, 5], ylimits=[-5, 5], numticks=1000, *args, **kwargs):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    levels = np.arange(-15.0, 10.5, 2.)
    ax.contour(X, Y, Z, levels=levels, *args, **kwargs)
    ax.set_yticks([])
    ax.set_xticks([])


if __name__ == '__main__':

    iters = 500
    noise_var = 0.1

    inputs = np.array([[-1.], [0.], [1.], [2.]])

    real_mean = np.array([0., 1., 0, 1.])
    r = np.array([[1.0, 2.0, 0., 0.],
                  [0., 1.0, 0., 1.],
                  [1.0, .0, 1., 1.],
                  [1., 1., 0., 1.]])
    real_cov = np.dot(r.T, r)

    layer_sizes = [1, 10, 10, 1]
    N_samples = 100

    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax1 = fig.add_subplot(221, frameon=False)
    #fig2 = plt.figure(figsize=(8, 8), facecolor='white')
    ax2 = fig.add_subplot(223, frameon=False)
    ax3 = fig.add_subplot(224, frameon=False)
    plt.ion()
    plt.show(block=False)

    def obj(params, t):
        return kl_estimate(params, N_samples, inputs, layer_sizes, real_mean, real_cov)

    def init_bnn_params(layer_sizes, scale):
        """initial mean and log std of q(w|phi) """
        _, N_weights = shapes_and_num(layer_sizes)
        mean = rs.randn(N_weights)
        log_std = np.zeros((N_weights,)) + scale
        return (mean, log_std)

    def plot_contours(ax, params):
        samples = sample_obs(params, N_samples, inputs, layer_sizes)
        y_mean, y_cov = np.mean(samples, axis=0), np.cov(samples.T)

        approx_pdf = lambda x: mvn.logpdf(x, y_mean, y_cov)
        real_pdf   = lambda x: mvn.logpdf(x, real_mean, real_cov)

        plot_isocontours(ax, approx_pdf, colors='r', label='approx')
        plot_isocontours(ax, real_pdf,   colors='b', label='true')


    def plot_lines(ax, params, inputs):
        n_functions = 10
        plot_inputs = np.array(list(np.linspace(-8, 8, num=200)) + list(inputs))
        plot_inputs.sort()
        ixs = []
        for i in list(inputs):
            ixs.append(np.where(plot_inputs == i)[0])

        f_bnn = sample_obs(params, n_functions, plot_inputs[:, None], layer_sizes, with_noist=False)
        ax.plot(plot_inputs, f_bnn.T, color='green')

        for i in range(n_functions):
            ax.scatter(inputs, [f_bnn[i, ix] for ix in ixs], marker='o')

    def plot_heatmap(ax, params):
        samples = sample_obs(params, N_samples, inputs, layer_sizes)
        _, y_cov = np.mean(samples, axis=0), np.cov(samples.T)
        ax.imshow(y_cov)

    def callback_kl(prior_params, iter, g):
        plot_lines(ax1, prior_params, inputs)
        plot_heatmap(ax2, prior_params)
        ax3.imshow(real_cov)
        print("Iteration {} KL {} ".format(iter, obj(prior_params, iter)))

        plt.draw()
        # plt.savefig(os.path.join(plotting_dir, 'contours_iteration_' + str(iter) + '.pdf'))
        plt.pause(1.0 / 400.0)
        ax1.cla()
        ax2.cla()
        ax3.cla()

    init_var_params = init_bnn_params(layer_sizes, scale=-1.5)
    prior_params = adam(grad(obj), init_var_params, step_size=0.05, num_iters=iters, callback=callback_kl)
