import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky
from kernels import covariance


def morph_bnn(layer_sizes, nonlinearity=np.tanh,
              noise_var=0.01, L2_reg=0.1, kernel="rbf"):

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
        """
        implements the forward pass of the bnn
        weights                                        dim = [N_weight_samples, N_weights]
        inputs                                         dim = [N_data]
        outputs                                        dim = [N_weight_samples, N_data,1]
        """

        inputs = np.expand_dims(inputs, 0)
        params = list(unpack_layers(weights))
        for W, b in params:
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)

        return outputs

    def logpost_l2(weights, inputs, targets):
        """
        computes the unnormalized log posterior of the weights
        using a standard normal prior p(w) = N(0,I)
        log p(w|D) = log p(D|w)+ log p(w)
        where D = {xi,yi}   i=1,...,N_data

        weights:                                        dim = [N_weight_samples, N_weights]
        inputs: xi                                      dim = [N_data]
        targets: yi in                                  dim = [N_data]

        compute 1) log_prior = log p(w)                 dim = [N_weights_samples]
                2) log_lik = log(D|w)                   dim = [N_weights_samples]

        log posterior = log p(D|w)+ log p(w)            dim = [N_weights_samples]
        """
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_var
        return log_prior + log_lik

    def log_gp_prior(bnn_weights, inputs):
        """
        computes: the expectation value of the log of the gp prior :
        E_{X~p(X)} [log p_gp(f)] where p_gp(f) = N(f|0,K) where f ~ p_BNN(f)
        we use the cholesky decomposition of K : L
        -0.5*E_{X~p(X)} [(L^-1f)^T(L^-1f)] + constants
        (we ignore constants for now as we are not optimizing the covariance hyperparams)

        param:
        inputs                                          dim = [N_data]
        bnn_weights = weights of the BNN                dim = [N_weights_samples, N_weights]

        K = the covariance matrix                       dim = [N_data, N_data]
        f_bnn output of a bnn                           dim = [N_weights_samples, N_data]
        L = Cholesky(K)                                 dim = dim K

        returns : E[log p_gp(f)]                        dim = [N_function_samples]
        """

        x = inputs
        f_bnn = predictions(bnn_weights, x)[:, :, 0].T
        K = covariance(x, x)+1e-7*np.eye(len(x))
        L = cholesky(K)
        a = solve(L, f_bnn).T
        log_gp = -0.5*np.mean(a**2, axis=1)

        # print("shapes | K^-1f {} | fbnn {} | K {} | log_gp_prior {} |".format(
        #      a.shape, f_bnn.shape, K.shape, log_gp.shape))

        return log_gp

    def log_post(weights, inputs, targets, prior_param):
        """
        computes the (unnormalized) log posterior of the weights using the learned prior
        log p(w|D) = log p(D|w)+ log p(w|phi)
        where D = {xi,yi}   i=1,...,N_data

        weights:                                        SHAPE = [N_weight_samples, N_weights]
        inputs: xi                                      SHAPE = [N_data]
        targets: yi in                                  SHAPE = [N_data]
        prior_param = phi :
        learned params of the prior                     SHAPE = [2*N_weights].
        unpacked into prior_mean                        SHAPE = [N_weights]
                  and prior_log_std                     SHAPE = [N_weights]

        compute 1) log_prior = log q(w|phi)             SHAPE = [N_weights_samples]
                2) log_lik = log(D|w)                   SHAPE = [N_weights_samples]

        log posterior = log p(D|w)+ log q(w|phi)        SHAPE = [N_weights_samples]
        """

        prior_mean, prior_log_std = unpack_params(prior_param)
        log_prior = -0.5*np.sum((weights-prior_mean)**2 / np.exp(prior_log_std), axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_var

        return log_prior + log_lik

    return N_weights, predictions, logpost_l2, log_gp_prior, log_post
