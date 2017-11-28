import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
rs = npr.RandomState(0)


def kl_inference(log_gp_prior, N_weights, N_samples):

    def unpack_params(params):
        mean, log_std = params[:N_weights], params[N_weights:]
        return mean, log_std

    def gaussian_entropy(samples):
        return np.log()


    def kl_objective(params, t):
        """
        Provides a stochastic estimate of the kl divergence
        kl= E_{p_BNN(f|phi)} [log p_BNN(f|phi) -log p_GP(f|theta)]
        mean, log_std                                           dim = [N_weights]
        samples                                                 dim = [N_samples, N_weights]
        kl                                                      dim = [1] """

        prior_mean, prior_log_std = unpack_params(params)
        samples = rs.randn(N_samples, N_weights) * np.exp(prior_log_std) + prior_mean
        kl = - np.mean(log_gp_prior(samples, t))
        return kl

    grad_kl = grad(kl_objective)

    return kl_objective, grad_kl, unpack_params


def vlb_inference(logprob, N_weights, N_samples):
    def unpack_params(params):
        mean, log_std = params[:N_weights], params[N_weights:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * N_weights * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def vlb_objective(params, t):
        """
        Provides a stochastic estimate of the variational lower bound
        ELBO = -E_q(w)[p(D|w)] - KL[q(w)|p(w)]

        params                                                  dim = [2*N_weights]
        mean, log_std                                           dim = [N_weights]
        samples                                                 dim = [N_samples, N_weights]
        elbo                                                    dim = [1]
        """
        mean, log_std = unpack_params(params)
        samples = rs.randn(N_samples, N_weights) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        return -lower_bound

    grad_vlb = grad(vlb_objective)

    return vlb_objective, grad_vlb, unpack_params


def gan_inference(gan_log_prob, N_weights, N_samples):

    def unpack_params(params):
        mean, log_std = params[:N_weights], params[N_weights:]
        return mean, log_std

    def gan_objective(bnn_params, d_params, t):
        """
        Provides a stochastic estimate of the gan objective
        V(G,D) = E_p_gp[log D(f)] + E_pbnn[log[ 1-D(f)]]

        params                                                  dim = [2*N_weights]
        mean, log_std                                           dim = [N_weights]
        samples                                                 dim = [N_samples, N_weights]
        """
        mean, log_std = unpack_params(bnn_params)
        bnn_weight_samples = rs.randn(N_samples, N_weights) * np.exp(log_std) + mean
        v = np.mean(gan_log_prob(bnn_weight_samples, d_params, t))
        return v

    grad_gan = grad(gan_objective, argnum=(0, 1))

    return gan_objective, grad_gan, unpack_params