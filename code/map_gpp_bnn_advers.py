import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten
from optimizers import adam_minimax
from kernels import covariance
from util import build_toy_dataset
from blackbox_svi import gan_inference
from autograd.numpy.linalg import solve, cholesky
import seaborn as sns
def rbf(x): return np.exp(-x**2)
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 0.5 * (np.tanh(x) + 1.0)

def morph_gan(bnn_layer_sizes,
              bnn_nonlinearity=rbf, nn_nonlinearity=relu):

    bnn_shapes = list(zip(bnn_layer_sizes[:-1], bnn_layer_sizes[1:]))
    n_bnn_weights = sum((m+1)*n for m, n in bnn_shapes)

    def logsigmoid(x):
        return x - np.logaddexp(0, x)

    def unpack_layers(weights):
        """ unpacks the bnn_weights into relevant tensor shapes """
        num_weight_sets = len(weights)
        for m, n in bnn_shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def bnn_predict(weights, inputs):
        """ implements the forward pass of the bnn
        weights  |  dim = [N_samples, N_weights]
        inputs   |  dim = [N_data]
        outputs  |  dim = [N_samples, N_data] """
        inputs = np.expand_dims(inputs, 0)
        bnn_params = list(unpack_layers(weights))
        for W, b in bnn_params:
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = bnn_nonlinearity(outputs)
        return outputs[:, :, 0]

    def sample_gp_prior(x, n_samples):
        """ Samples from the gp prior x = inputs with shape [N_data]
        returns : samples from the gp prior [N_samples, N_data] """
        x = np.ravel(x)
        n_data = len(x)
        K = covariance(x[:, None], x[:, None])
        L = cholesky(K + 1e-4 * np.eye(n_data))
        e = np.random.normal(size=(n_data, n_samples))
        f_gp_prior = np.dot(L, e)
        return f_gp_prior.T

    def batch_normalize(activations):  # not used for regression exp
        mbmean = np.mean(activations, axis=0, keepdims=True)
        s = (np.std(activations, axis=0, keepdims=True) + 1)
        return (activations - mbmean) / s

    def nn_predict(nn_params, inputs):
        """Params = list of (weights, bias)
           inputs = N x D matrix."""
        for W, b in nn_params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = nn_nonlinearity(outputs)
        outW, outb = nn_params[-1]
        outputs = np.dot(inputs, outW) + outb
        return outputs

    def gan_log_prob(bnn_weights, d_params, n_data, n_samples):
        x = np.random.normal(size=(n_data, 1))  # sample X ~ P(X)
        bnn_samples = bnn_predict(bnn_weights, x)  # sample f ~ P_bnn(f)
        gp_samples = sample_gp_prior(x, n_samples)  # sample f ~ P_gp(f)
        logprobs_bnn = logsigmoid(nn_predict(d_params, bnn_samples))
        logprobs_gp = logsigmoid(nn_predict(d_params, gp_samples))
        log_prob = logprobs_gp - logprobs_bnn
        return log_prob

    return n_bnn_weights, gan_log_prob, bnn_predict, nn_predict, sample_gp_prior


def init_nn_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) for each nn layer"""
    return [(scale * rs.randn(m, n), scale * rs.randn(n))
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_bnn_params(num_weights, scale=-2):
    """packs together mean and log std of variational approx
       to posterior which is a diagonal gaussian """
    init_mean = rs.randn(num_weights)
    init_log_std = scale * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])
    return init_var_params

if __name__ == '__main__':
    rs = npr.RandomState(0)
    data = "xsinx"
    n_data = 200
    samples = 15
    save_plots = False
    plot_during = True

    gen_layer_sizes = [1, 20, 20, 1]
    dsc_layer_sizes = [n_data, 50, 25, 1]

    # Training parameters
    param_scale = 0.01
    batch_size = 5
    num_epochs = 200

    step_size_max = 0.01
    step_size_min = 0.01

    n_bnn_weights, gan_log_prob, \
    bnn_predict, nn_predict, sample_gp = morph_gan(bnn_layer_sizes=gen_layer_sizes)

    log_prob = lambda bnn_w, d_param, t: gan_log_prob(bnn_w, d_param, n_data, samples)
    gan_objective, grad_gan, unpack_params = gan_inference(log_prob, n_bnn_weights, samples)

    # set up fig
    if plot_during:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(gen_params, dsc_params, iter, gen_gradient, dsc_gradient):
        # Sample functions from prior f ~ p(f|phi) or p(f|varphi)
        n_samples = 3
        plot_inputs = np.linspace(-8, 8, num=100)
        mean, log_std = unpack_params(gen_params)
        sample_weights = rs.randn(n_samples, n_bnn_weights) * np.exp(log_std) + mean
        f_bnn = bnn_predict(sample_weights, plot_inputs[:, None]).T  # f ~ p_bnn (f)
        f_gp = sample_gp(plot_inputs, n_samples).T                   # f ~ p_gp  (f)

        # Plot functions.
        if plot_during:
            plt.cla()
            # ax.plot(x.ravel(), y.ravel(), 'ko')
            ax.plot(plot_inputs, f_bnn, color='r')
            ax.plot(plot_inputs, f_gp, color='g')
            ax.set_ylim([-10, 10])
            plt.draw()
            plt.pause(1.0/40.0)

        print("Iteration {} ".format(iter))


    # INITIALIZE THE PARAMETERS
    init_gen_params = init_bnn_params(n_bnn_weights, scale=-1.5)
    init_dsc_params = init_nn_params(param_scale, dsc_layer_sizes)

    optimized_params = adam_minimax(grad_gan, init_gen_params, init_dsc_params,
                                    step_size_max=step_size_max, step_size_min=step_size_min,
                                    num_iters=num_epochs, callback=callback)
