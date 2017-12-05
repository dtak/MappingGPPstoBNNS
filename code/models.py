import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky, det
from autograd import grad
from kernels import kernel_dict
rs = npr.RandomState(0)




def map_gpp_bnn(layer_sizes, nonlinearity=np.tanh,
                n_data=100, N_samples=20, kernel="rbf"):

    covariance = kernel_dict[kernel]

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m+1)*n for m, n in shapes)

    def unpack_params(params):
        mean, log_std = params[:N_weights], params[N_weights:]
        return mean, log_std

    def unpack_layers(weights):
        """ unpacks the weights into relevant tensor shapes """
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """ implements the forward pass of the bnn
        weights | dim = [N_weight_samples, N_weights]
        inputs  | dim = [N_data]
        outputs | dim = [N_weight_samples, N_data,1] """

        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def init_bnn_params(N_weights, scale=-5):
        """initial mean and log std of q(w) ~ N(w|mean,std)"""
        mean = rs.randn(N_weights)
        log_std = scale * np.ones(N_weights)
        return np.concatenate([mean, log_std])

    def sample_gpp(x, n_samples):
        """ Samples from the gp prior x = inputs with shape [N_data]
        returns : samples from the gp prior [N_data, N_samples] """
        x = np.ravel(x)
        n_data = len(x)
        s =  1e-7 * np.eye(n_data)
        K = covariance(x[:, None], x[:, None])+s
        L = cholesky(K)
        e = rs.randn(n_data, n_samples)
        return np.dot(L, e)

    def sample_bnn(x, n_samples, params=None):
        """samples functions from a bnn"""
        if params is not None:  # sample learned learned prior var params
            mean, log_std = unpack_params(params)
            bnn_weights = rs.randn(n_samples, N_weights) * np.exp(log_std) + mean
        else:  # sample standard normal prior weights
            bnn_weights = rs.randn(n_samples, N_weights)
        return predictions(bnn_weights, x[:, None])[:, :, 0].T


    def log_gp_prior(f_bnn, x, t):
        """ computes: the expectation value of the log of the gp prior :
        E_{X~p(X)} [log p_gp(f)] where p_gp(f) = N(f|0,K) where f ~ p_BNN(f)
        = -0.5 * E_{X~p(X)} [ (L^-1f)^T(L^-1f) ] + const; K = LL^T (cholesky decomposition)
        (we ignore constants for now as we are not optimizing the covariance hyperparams)

        bnn_weights                   |  dim = [N_weights_samples, N_weights]
        K = covariance/Kernel matrix  |  dim = [N_data, N_data] ; dim L = dim K
        f_bnn output of a bnn         |  dim = [N_data, N_weights_samples]
        returns : E[log p_gp(f)]      |  dim = [N_function_samples] """

        s= 1e-6*np.eye(len(x))
        K = covariance(x, x) +s                     # shape [N_data, N_data]
        L = cholesky(K)+s                                 # shape K = LL^T
        a = solve(L, f_bnn)                             # shape = shape f_bnn (L^-1 f_bnn)
        log_gp = -0.5*np.mean(a**2, axis=0)             # Compute E_{X~p(X)}
        return log_gp

    def entropy_estimate(f_bnn):
        """ estimate of the entropy og p_bnn(f )
        given by H[p_bnn(f)] = E_{p_bnn(f} [-log p_bnn(f)]
        f_bnn shape [N_data, N_weights_samples]      """
        n_data, n_samples = f_bnn.shape

        entropy = -0.001

        # sample diagonal covariance
        # var_moment = np.var(f_bnn, axis=0)  # shape = [N_data]
        # entropy = 0.008*np.sum(np.log(var_moment))

        # sample covariance matrix using the scatter matrix
        # f_bnn=f_bnn/np.max(f_bnn)
        # C = np.eye(n_samples)-np.ones((n_samples, n_samples))/n_samples
        # S = np.dot(np.dot(f_bnn, C), f_bnn.T)/n_samples
        # print("shape", S.shape)
        # entropy = 0.5*np.log(det(S))

        return entropy

    def kl_objective(params, t):
        """
        Provides a stochastic estimate of the kl divergence
        kl= E_{p_BNN(f|phi)} [log p_BNN(f|phi) -log p_GP(f|theta)]

        mean, log_std  |  dim = [N_weights]
        samples        |  dim = [N_samples, N_weights]
        f_bnn          |  dim = [N_data, N_weights_samples]

        kl             |  dim = [1] """

        prior_mean, prior_log_std = unpack_params(params)
        bnn_weights = rs.randn(N_samples, N_weights) * np.exp(prior_log_std) + prior_mean
        x = np.random.uniform(low=-10, high=10, size=(n_data, 1))
        f_bnn = predictions(bnn_weights, x)[:, :, 0].T  # shape [N_data, N_weights_samples]
        kl = - entropy_estimate(f_bnn) - np.mean(log_gp_prior(f_bnn, x,  t))
        return kl

    grad_kl = grad(kl_objective)

    return N_weights, predictions, unpack_params, \
           init_bnn_params, sample_bnn, sample_gpp, \
           kl_objective, grad_kl


# BAYESIAN NEURAL NET


def construct_bnn(layer_sizes, nonlinearity=np.tanh,
                  noise_var=0.1, L2_reg=0.1,
                  N_samples=20):

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m+1)*n for m, n in shapes)

    def init_bnn_params(N_weights, scale=-5):
        """initial mean and log std of q(w|phi) """
        mean = rs.randn(N_weights)
        log_std = scale * np.ones(N_weights)
        return np.concatenate([mean, log_std])

    def unpack_params(params):
        mean, log_std = params[:N_weights], params[N_weights:]
        return mean, log_std

    def unpack_layers(weights):
        """ unpacks the weights into relevant tensor shapes """
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """ implements the forward pass of the bnn
        weights | dim = [N_weight_samples, N_weights]
        inputs  | dim = [N_data]
        outputs | dim = [N_weight_samples, N_data,1] """

        inputs = np.expand_dims(inputs, 0)
        params = list(unpack_layers(weights))
        for W, b in params:
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def sample_bnn(x, n_samples, params=None):
        """samples functions from a bnn"""
        if params is not None:  # sample learned learned prior var params
            mean, log_std = unpack_params(params)
            bnn_weights = rs.randn(n_samples, N_weights) * np.exp(log_std) + mean
        else:  # sample standard normal prior weights
            bnn_weights = rs.randn(n_samples, N_weights)
        return predictions(bnn_weights, x[:, None])[:, :, 0].T

    def log_post(weights, inputs, targets, prior_param=None):
        """ computes the unnormalized log posterior of the weights using
        the learned prior p(w|phi*) or a standard normal prior p(w) = N(0,I)
        log p(w|D) = log p(D|w)+ log p(w|phi) where D = {xi,yi}   i=1, ..., N_data

        weights:                                    |  dim = [N_weight_samples, N_weights]
        D = (inputs, targets) = (xi,yi)             |  dim = ([N_data], [N_data])

        prior_param = learned params of the prior   |  dim = [2*N_weights]
        unpacked into prior_mean and prior_log_std  |  dim = [N_weights]

        log_prior = log q(w|phi)                    |  dim = [N_weights_samples]
        log_lik = log(D|w)                          |  dim = [N_weights_samples] """

        if prior_param is None:
            log_prior = -L2_reg * np.sum(weights ** 2, axis=1)
        else:
            prior_mean, prior_log_std = unpack_params(prior_param)
            log_prior = -0.5*np.sum((weights-prior_mean)**2 / np.exp(prior_log_std), axis=1)

        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_var

        return log_prior + log_lik

    def gaussian_entropy(log_std):
        return 0.5 * N_weights * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def vlb_objective(params, log_prob, t):
        """ Provides a stochastic estimate of the variational lower bound
        ELBO = -E_q(w)[p(D|w)] - KL[q(w)|p(w)]

        params                                                  dim = [2*N_weights]
        mean, log_std                                           dim = [N_weights]
        samples                                                 dim = [N_samples, N_weights]
        lower bound                                             dim = [1] """
        mean, log_std = unpack_params(params)
        samples = rs.randn(N_samples, N_weights) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(log_prob(samples, t))
        return -lower_bound

    return N_weights, init_bnn_params, predictions,sample_bnn, log_post, unpack_params, vlb_objective
