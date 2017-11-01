import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky
from autograd.optimizers import adam, sgd
from blackbox_svi import kl_inference, elbo_inference
from util import covariance, build_toy_dataset

rs = npr.RandomState(0)


def morph_bnn(layer_sizes, nonlinearity=np.tanh, noise_var=0.01):

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        """ unpacks the weights into relevant tensor shapes """
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """
        implements the forward pass of the bnn
        weights                                                 dim = [N_weight_samples, N_weights]
        inputs                                                  dim = [N_data]
        outputs                                                 dim = [N_weight_samples, N_data,1]
        """

        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)

        return outputs

    def log_gp_prior(bnn_weights, inputs):
        """
        the log of the gp prior : log p_gp(f) = N(f|0,K) where f ~ p_BNN(f)

        bnn_weights = weights of the BNN                    dim = [N_weights_samples, N_weights]

        K = the covariance matrix                           dim = [N_inputs, N_inputs]
        f_bnn output of a bnn                               dim = [N_weights_samples, N_inputs]
        L=cholesky(K)
        computes log p_gp (f) = -0.5*E[(L^-1f)^T(L^-1f)] + constants

        returns : log p_gp(f)                               dim = [N_fsamples]
        """

        x = inputs
        f_bnn = predictions(bnn_weights, x)[:, :, 0].T
        K = covariance(x, x)+noise_var*np.eye(len(x))
        L = cholesky(K)
        a = solve(L, f_bnn).T
        log_gp = -0.5*np.mean(a**2, axis=1)

        # print("shapes | K^-1f {} | fbnn {} | K {} | log_gp_prior {} |".format(
        #      a.shape, f_bnn.shape, K.shape, log_gp.shape))

        return log_gp

    def log_post(weights, inputs, targets, prior_param):
        """
        computes the log posterior of the weights
        log p(w|D) = log p(D|w)+ log q(w|phi)
        where D = {xi,yi}   i=1,...,N_data

        weights:                                        SHAPE = [N_weight_samples, N_weights]
        inputs: xi                                      SHAPE = [N_data]
        targets: yi in                                  SHAPE = [N_data]
        prior_param = phi :
        learned params of the diagonal
        gaussian approximation to p_GP(f)               SHAPE = [2*N_weights].
        unpacked into prior_mean                        SHAPE = [N_weights]
                  and prior_log_std                     SHAPE = [N_weights]

        compute 1) log_prior = log q(w|phi)             SHAPE = [N_weights_samples]
                2) log_lik = log(D|w)                   SHAPE = [N_weights_samples]

        log posterior = log p(D|w)+ log q(w|phi)        SHAPE = [N_weights_samples]
        """

        # LEARNED PRIOR
        prior_mean, prior_log_std = unpack_params(prior_param)
        log_prior = -0.5*np.sum((weights-prior_mean)**2 / np.exp(prior_log_std), axis=1)

        # L2 PRIOR
        #log_prior = -0.1 * np.sum(weights**2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_var

        return log_prior + log_lik

    return num_weights, predictions, log_gp_prior, log_post




if __name__ == '__main__':
    rs = npr.RandomState(0)
    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)

    N_data = 70
    samples = 20
    iters_1 = 20
    iters_2 = 100

    num_weights, predictions, log_p_gp, log_post = morph_bnn(layer_sizes=[1, 20, 20, 1],
                                                             nonlinearity=rbf)

    inputs, targets = build_toy_dataset(n_data=N_data)
    log_gp_prior = lambda w, t: log_p_gp(w, inputs)
    kl, grad_kl, unpack_params = kl_inference(log_gp_prior,
                                              N_weights=num_weights,
                                              N_samples=samples)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback1(params, t, g):
        # Sample functions from posterior.
        mean, log_std = unpack_params(params)
        sample_weights = rs.randn(10, num_weights) * np.exp(log_std) + mean
        plot_inputs = np.linspace(-8, 8, num=400)
        outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))

        # SAMPLE FROM THE GP PRIOR
        #L = cholesky(covariance(plot_inputs[:,None], plot_inputs[:,None]) + noise_var*np.eye(400))
        #f_gpprior = np.dot(L, np.random.normal(size=(400, 10)))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx')
        ax.plot(plot_inputs, outputs[:, :, 0].T, color='g')
        ax.set_title("learning the prior")
        print("Iteration {} | kl {}".format(t, kl(params,t)))

        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)

    init_mean = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    print("num weights :", num_weights)
    print("shape var params : ", init_var_params.shape)

    prior_params = adam(grad_kl, init_var_params, step_size=0.1,
                                                  num_iters=iters_1,
                                                  callback=callback1)

    print("Finished Optimizing the prior")

    def callback2(params, t, g):
        # Sample functions from posterior.
        rs = npr.RandomState(0)
        mean, log_std = unpack_params(params)
        sample_weights = rs.randn(10, num_weights) * np.exp(log_std) + mean
        plot_inputs = np.linspace(-8, 8, num=400)
        outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx')
        ax.plot(plot_inputs, outputs[:, :, 0].T, color='g')
        ax.set_title("optimizing elbo")
        print("Iteration {} | elbo {}".format(t, -elbo(params, t)))

        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)

    init_mean = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])

    prior_params = adam(grad_kl, init_var_params, step_size=0.1,
                        num_iters=iters_1, callback=callback1)

    #prior_params = init_var_params
    #print(var_prior_params.shape)
    #print(prior_params)

    log_posterior = lambda weights, t: log_post(weights, inputs, targets, prior_params)
    elbo, grad_elbo, unpack_params = elbo_inference(log_posterior, N_weights=num_weights,
                                                    N_samples=samples)

    print("OPTIMIZING ELBO ")
    var_params = adam(grad_elbo, init_var_params,
                              step_size=0.1, num_iters=iters_2, callback=callback2)