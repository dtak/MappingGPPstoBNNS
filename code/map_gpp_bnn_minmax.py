import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.numpy.linalg import solve, cholesky, det
from autograd.misc.optimizers import adam
from autograd import grad
from optimizers import adam_minimax as adam
import kernels

# choose kernel here ; replace rbf with whatever
covariance = kernels.kernel_rbf
rs = npr.RandomState(0)


def map_gpp_bnn(layer_sizes, nonlinearity=np.tanh,
                n_data=200, N_samples=10,
                L2_reg=0.1, noise_var=0.1):

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m+1)*n for m, n in shapes)

    def unpack_params(params):
        mean, log_std = params[:N_weights], params[N_weights:]
        return mean, log_std

    def unpack_layers(weights):
        """ iterable that unpacks the weights into relevant tensor shapes for each layer"""
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """ implements the forward pass of the bnn
        weights | dim = [N_weight_samples, N_weights]
        inputs  | dim = [N_data]
        outputs | dim = [N_weight_samples, N_data, 1] """

        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def sample_gpp(x, n_samples):
        """ Samples from the gp prior x = inputs with shape [N_data]
        returns : samples from the gp prior [N_data, N_samples] """
        x = np.ravel(x)
        n_data = len(x)
        K = covariance(x[:, None], x[:, None])
        L = cholesky(K + 1e-7 * np.eye(n_data))
        e = rs.randn(n_data, n_samples)
        return np.dot(L, e)

    def log_gp_prior(y_bnn, x):
        """ computes: the expectation value of the log of the gp prior :
        E [ log p_gp(f) ] where p_gp(f) = N(f|0,K) where f ~ p_BNN(f)
        = -0.5 * E [ (L^-1f)^T(L^-1f) ] + const; K = LL^T (cholesky decomposition)
        (we ignore constants for now as we are not optimizing the covariance hyper-params)

        bnn_weights                   |  dim = [N_weights_samples, N_weights]
        K = covariance/Kernel matrix  |  dim = [N_data, N_data] ; dim L = dim K
        y_bnn output of a bnn         |  dim = [N_data, N_weights_samples]
        returns : E[log p_gp(y)]      |  dim = [N_function_samples] """

        K = covariance(x, x)+noise_var*np.eye(len(x))   # shape [N_data, N_data]
        L = cholesky(K)                                 # K = LL^T ; shape L = shape K
        a = solve(L, y_bnn)                             # a = L^-1 y_bnn ; shape L^-1 y_bnn =
        log_gp = -0.5*np.mean(a**2, axis=0)             # Compute E [|a|^2]
        return log_gp

    def log_prob(weights, inputs, targets):
        """ computes log p(y,w) = log p(y|w)+ log p(w)
            with a p(w) = N(w|0,I)
        weights:                  |  dim = [N_weight_samples, N_weights]
        preds = f                 |  dim = [N_weight_samples, N_data, 1]
        targets = y               |  dim = [N_data]

        log_prior = log p(w)      |  dim = [N_weights_samples]
        log_lik = log(y|w)        |  dim = [N_weights_samples] """


        log_prior = -L2_reg * np.sum(weights ** 2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_var

        return log_prior + log_lik

    def gaussian_entropy(log_std):
        return 0.5 * N_weights * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def elbo(var_param, x, y):
        """ Provides a stochastic estimate of the evidence lower bound
        ELBO = E_r(w) [log p(y,w)-r(w)]

        params          |   dim = [2*N_weights]
        mean, log_std   |   dim = [N_weights]
        samples         |   dim = [N_samples, N_weights]
        returns : ELBO  |   dim = [1] """

        mean, log_std = unpack_params(var_param)
        ws = rs.randn(N_samples, N_weights) * np.exp(log_std) + mean  # sample weights from r(w)
        return gaussian_entropy(log_std) + np.mean(log_prob(ws, x, y))  # ELBO

    def entropy(thetas, ys, x):
        """ creates an array of log p(y) for each y in ys
        which are estimated by using the ELBO
        log p(y) => E_r(w) [ log p(y,w)-log r(w)]
        ys has shape [y_samples, N_data] """

        #  get E_r(w)[p(y,w) - r(w)] for each w, y
        elbos = np.array([elbo(theta, x, y) for theta, y in zip(thetas, ys)])

        return elbos

    def kl_objective(params_phi, params_theta, t):
        """
        Provides a stochastic estimate of the kl divergence
        kl[p(y)|p_GP(y)] = E_p(y) [log p(y) -log p_gp(y)]
                         = -H[ p(y) ] -E_p(y) [log p_gp(y)]
        using :
        params_phi        dim = [2*N_weights]
        params_theta      list of [2*N_weights] : the var params of each r(w)

        phi_mean, phi_log_std  |  dim = [N_weights]

        w_phi        |  dim = [N_samples, N_weights]
        y_bnn        |  dim = [N_data, N_weights_samples]

        kl             |  dim = [1] """

        phi_mean, phi_log_std = unpack_params(params_phi)
        w_phi = rs.randn(N_samples, N_weights) * np.exp(phi_log_std) + phi_mean
        x = np.random.uniform(low=-10, high=10, size=(n_data, 1))  # X ~ p(X)

        f_bnn = predictions(w_phi, x)[:, :, 0].T  # shape [N_data, N_weights_samples] f ~ p(f)
        y_bnn = f_bnn + 3*noise_var*rs.randn(n_data, N_samples)  # y = f + e ; y ~ p(y)

        # use monte carlo to approximate H[p(y)] = E_p(y)[ log p(y)] and E_p(y) [log p_gp(y)]
        kl_div = np.mean(entropy(params_theta, y_bnn.T, x)) - np.mean(log_gp_prior(y_bnn, x))

        return kl_div  # the KL

    grad_kl = grad(kl_objective, argnum=(0, 1))

    return N_weights, predictions, sample_gpp, unpack_params, kl_objective, grad_kl

if __name__ == '__main__':
    rs = npr.RandomState(0)

    # activation functions
    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)
    sigmoid = lambda x: 0.5*(np.tanh(x)-1)

    exp_num = 26
    data = 'xsinx'  # or expx or cosx

    samples = 10
    iters_1 = 500
    noise_var = 0.1

    plot_during = True

    num_weights, bnn_predict, sample_gpp, unpack_params, \
    kl, grad_kl = map_gpp_bnn(layer_sizes=[1, 20, 20, 1], nonlinearity=rbf,
                              n_data=200, N_samples=15)

    def init_bnn_params(N_weights, scale=-5):
        """initial mean and log std of q(w|phi) """
        mean = rs.randn(N_weights)
        log_std = scale * np.ones(N_weights)
        return np.concatenate([mean, log_std])


    def init_thetas_params(N_weights, N_samples, scale=-5):
        """list of initial mean and log std of r(w|theta) for each y"""
        mean = rs.randn(N_weights)
        log_std = scale * np.ones(N_weights)
        return [np.concatenate([mean, log_std])]*N_samples

    def sample_bnn(x, n_samples, params=None):
        """samples functions from a bnn"""
        n_data = x.shape[0]
        if params is not None:  # sample using reparameterization trick
            mean, log_std = unpack_params(params)
            bnn_weights = rs.randn(n_samples, num_weights) * np.exp(log_std) + mean
        else:  # sample standard normal prior weights
            bnn_weights = rs.randn(n_samples, num_weights)
        f_bnn = bnn_predict(bnn_weights, x[:, None])[:, :, 0].T
        return f_bnn  # shape [n_data, n_samples]



    if plot_during:
        f, ax = plt.subplots(3, sharex=True)
        plt.ion()
        plt.show(block=False)

    def callback_kl(phi, theta, iter, gphi, gtheta):
        # Sample functions from priors f ~ p(f)
        n_samples = 3
        plot_inputs = np.linspace(-8, 8, num=100)

        f_bnn_gpp = sample_bnn(plot_inputs, n_samples, phi)    # f ~ p_bnn (f|phi)
        f_bnn     = sample_bnn(plot_inputs, n_samples)                  # f ~ p_bnn (f)
        f_gp      = sample_gpp(plot_inputs, n_samples)                  # f ~ p_gp  (f)

        # Plot samples of functions from the bnn and gp priors.
        if plot_during:
            for axes in ax: axes.cla()
            # ax.plot(x.ravel(), y.ravel(), 'ko')
            ax[0].plot(plot_inputs, f_gp, color='green')
            ax[1].plot(plot_inputs, f_bnn_gpp, color='red')
            ax[2].plot(plot_inputs, f_bnn, color='blue')
            #ax[0].set_ylim([-5, 5])
            #ax[1].set_ylim([-5, 5])
            #ax[2].set_ylim([-5, 5])

            plt.draw()
            plt.pause(1.0/40.0)

        print("Iteration {} KL {} ".format(iter, kl(phi, theta, iter)))

    # ----------------------------------------------------------------

    # Initialize the variational params HERE
    # the functions drawn from the optimized bnn prior are heavily
    # dependent on which initialization scheme used here

    init_phis = init_bnn_params(num_weights, scale=-0.5)
    init_thetas = init_thetas_params(num_weights, samples, scale=-5)

    # ---------------------- MINIMIZE THE KL --------------------------

    prior_params = adam(grad_kl, init_phis, init_thetas,
                        step_size_max=0.01, step_size_min=0.1,
                        num_iters=iters_1, callback=callback_kl)