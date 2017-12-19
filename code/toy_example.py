import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.numpy.linalg import solve, cholesky, det
from autograd.misc.optimizers import adam
from autograd import grad
from models import construct_bnn
import kernels
from numpy import mgrid
from scipy.stats import multivariate_normal
# from autograd.scipy.stats import multivariate_normal


# choose kernel here ; replace rbf with
covariance = kernels.kernel_rbf
rs = npr.RandomState(0)


def is_pos_def(K):
    # not very elegant but it works
    try:
        np.linalg.cholesky(K)
        return True
    except:
        return False


def reg_mat(K):
    i = 0
    while not is_pos_def(K):
        K += 10 * i * 1e-10 * np.eye(K.shape[0])
        i += 1
    return K

def map_gpp_bnn(layer_sizes, nonlinearity=np.tanh,
                n_data=200, N_samples=20, noise_var=0.1):

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m+1)*n for m, n in shapes)

    def unpack_params(params):
        mean, log_chol = params[:N_weights], params[N_weights:]
        log_chol = np.reshape(log_chol, (N_weights, N_weights))
        return mean, log_chol

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

    # def log_gp_prior(y_bnn, x):
    #     """ computes: the expectation value of the log of the gp prior :
    #     E [ log p_gp(f) ] where p_gp(f) = N(f|0,K) where f ~ p_BNN(f)
    #     = -0.5 * E [ (L^-1f)^T(L^-1f) ] + const; K = LL^T (cholesky decomposition)
    #     (we ignore constants for now as we are not optimizing the covariance hyper-params)
    #
    #     bnn_weights                   |  dim = [N_weights_samples, N_weights]
    #     K = covariance/Kernel matrix  |  dim = [N_data, N_data] ; dim L = dim K
    #     y_bnn output of a bnn         |  dim = [N_data, N_weights_samples]
    #     returns : E[log p_gp(y)]      |  dim = [N_function_samples] """
    #
    #     K = covariance(x, x)+noise_var*np.eye(len(x))   # shape [N_data, N_data]
    #     L = cholesky(K)                                 # K = LL^T ; shape L = shape K
    #     a = solve(L, y_bnn)                             # a = L^-1 y_bnn ; shape L^-1 y_bnn =
    #     log_gp = -0.5*np.mean(a**2, axis=0)             # Compute E [|a|^2]
    #     return log_gp

    # def entropy_estimate_approx_inference(y_bnn, x):
    #     """ estimate of the entropy of p(y) given by
    #     H[p_(y)] = E_{p(y} [log p(y)] = sum_y log p(y)
    #     where each log p(y) = max E_r(w) [ log p(y,w)-log r(w)]
    #     y_bnn shape [N_data, N_weights_samples]      """
    #
    #     _, y_samples = y_bnn.shape
    #     sum_log_p_y = 0
    #
    #     print("est H[p(y)]")
    #
    #     for y in y_bnn.T:
    #         print(y.shape)
    #         # construct approximate inference functions using the bnn
    #         n_weights_r, init_bnn_r_params, _, log_post, _, vlb_objective = \
    #             construct_bnn(layer_sizes, rbf)
    #
    #         # log p(y|w) + log p(w) where log p(y|w) = Sum_i N(y_i| f(x_i), noise_var*I )
    #         log_posterior = lambda w, t: log_post(w, x, y, t)
    #
    #         # use E_r[log p(y,w) - log r(w)] as estimate of log p(y)
    #         log_p_y = lambda params, t: vlb_objective(params, log_posterior, t)
    #
    #         # get the gradient of E_r [ log p(y,w)-log r(w)]
    #         grad_log_p_y = grad(log_p_y)
    #
    #         # initialize the variational params of r(w)
    #         init_var_r_params = init_bnn_params(n_weights_r)
    #
    #         # optimize them to extract an approximation to log p(y)
    #         var_params = adam(grad_log_p_y, init_var_r_params, step_size=0.1, num_iters=100)
    #         entropy = -log_p_y(var_params, t=20)  # sum the estimates
    #     print(entropy)
    #
    #     # use monte carlo to approximate H[p(y)] = E_p(y)[ log p(y)]
    #     entropy = sum_log_p_y/y_samples
    #     return entropy
    #
    # def entropy_estimate_kde(y_bnn, x):
    #
    #     def k(x, xp, h):
    #         d = x[:, None] - xp[None, :]
    #         return np.exp(-0.5 * np.sum(d/h ** 2, axis=2))/6.28
    #
    #     h = 2
    #     K = k(y_bnn.T, y_bnn.T, h=h)
    #     log_p_y = np.log(np.mean(K/h, axis=1))
    #     entropy = np.mean(log_p_y)
    #
    #     return -entropy

    def entropy_estimate_moments(y_bnn, x):
        centred_y = y_bnn - np.mean(y_bnn, axis=1)[:, None]
        cov = np.dot(centred_y, centred_y.T)/y_bnn.shape[1]
        return 0.5*np.log(np.linalg.det(2 * np.pi * np.exp(1) * reg_mat(cov)))

    def log_gauss(y_bnn, x):
        def lg(x, mu, cov):
            d = -0.5 * np.log(np.linalg.det(2. * np.pi * cov)) - \
                0.5 * np.trace(np.dot(np.dot((x.T - mu), np.linalg.inv(cov)), x - mu.T))
            return d

        real_mean = np.array([[0., 1.]])
        r = np.array([[1.0, 2.0], [0., 1.]])
        real_cov = np.dot(r.T, r)
        return lg(y_bnn, real_mean, real_cov)

    def sample_weights(params, N_samples):
        prior_mean, prior_log_chol = unpack_params(params)
        bnn_weights = np.dot(rs.randn(N_samples, N_weights), np.exp(prior_log_chol)) + prior_mean[None, :]
        return bnn_weights

    def sample_obs(params, N_samples):
        bnn_weights = sample_weights(params, N_samples)
        x = np.array([[0.], [1.]])

        f_bnn = predictions(bnn_weights, x)[:, :, 0].T  # shape [N_data, N_weights_samples]
        y_bnn = f_bnn + 3*noise_var*rs.randn(2, N_samples)  # y = f + e
        return x, y_bnn

    def kl_objective(params, t):
        """
        Provides a stochastic estimate of the kl divergence
        kl[p(y)|p_GP(y)] = E_p(y) [log p(y) -log p_gp(y)]
                         = -H[ p(y) ] + -E_p(y) [log p_gp(y)]

        mean, log_std  |  dim = [N_weights]
        samples        |  dim = [N_samples, N_weights]
        y_bnn          |  dim = [N_data, N_weights_samples]

        kl             |  dim = [1] """

        x, y_bnn = sample_obs(params, N_samples)
        kl_div = entropy_estimate_moments(y_bnn, x) + np.mean(log_gauss(y_bnn, x))
        return -kl_div  # the KL

    grad_kl = grad(kl_objective)

    def moment_match_outputs(params):
        x, y_bnn = sample_obs(params, N_samples)
        y_mean = np.mean(y_bnn, axis=1)
        y_centred = y_bnn - y_mean[:, None]
        y_cov = (1./y_bnn.shape[0]) * np.dot(y_centred, y_centred.T)
        return y_mean, y_cov

    return N_weights, predictions, sample_gpp, unpack_params, kl_objective, grad_kl, moment_match_outputs, \
           sample_weights

if __name__ == '__main__':
    rs = npr.RandomState(0)

    # activation functions
    rbf = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.)
    sigmoid =lambda x: 0.5*(np.tanh(x)-1)

    exp_num = 26
    data = 'xsinx'  # or expx or cosx
    N_data = 70
    samples = 50
    iters_1 = 500
    noise_var = 0.1

    plot_during = True

    num_weights, bnn_predict, sample_gpp, unpack_params, \
    kl, grad_kl, moment_match_outputs, sample_weights = map_gpp_bnn(layer_sizes=[1, 2, 1], nonlinearity=rbf,
                              n_data=N_data, N_samples=samples)

    def init_bnn_params(N_weights, scale=-5):
        """initial mean and log std of q(w|phi) """
        mean = rs.randn(N_weights)
        log_chol = scale * np.ones((N_weights, N_weights))
        return np.concatenate([mean, log_chol.flatten()])

    def sample_bnn(x, n_samples, params=None):
        """samples functions from a bnn"""
        n_data = x.shape[0]
        if params is not None:  # sample learned prior var params
            # mean, log_chol = unpack_params(params)
            # bnn_weights = rs.randn(n_samples, num_weights) * np.exp(log_std) + mean
            bnn_weights = sample_weights(params, n_samples)
        else:  # sample standard normal prior weights
            bnn_weights = rs.randn(n_samples, num_weights)
        f_bnn = bnn_predict(bnn_weights, x[:, None])[:, :, 0].T
        # y_bnn = f_bnn + 3*noise_var*rs.randn(n_data, n_samples)    # ADD Gaussian noise
        return f_bnn  # shape [n_data, n_samples]


    def plot_contours(prior_params):
        y_mean, y_cov = moment_match_outputs(params=prior_params)
        plt.figure(1)
        real_mean = np.array([0., 1.])
        r = np.array([[1.0, 2.0], [0., 1.]])
        real_cov = np.dot(r.T, r)

        # this is because multivariate normal expects a vector and my objects are matrices.
        x, y = mgrid[-5: 5:.01,
                     -5: 5:.01]

        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x;
        pos[:, :, 1] = y
        q_cov = reg_mat(y_cov)
        approx_pdf = multivariate_normal(y_mean, y_cov)
        real_pdf = multivariate_normal(real_mean, real_cov)

        plt.contour(x, y, approx_pdf.pdf(pos), colors='r', label='approx')
        plt.contour(x, y, real_pdf.pdf(pos), colors='b', label='true')
        # plt.savefig(os.path.join(plotting_dir, 'contours_iteration_' + str(iter) + '.pdf'))

        plt.show()
        plt.pause(1.0 / 40.0)
        plt.clf()


    def plot_lines(prior_params):
        f, ax = plt.subplots(3, sharex=True)
        plt.ion()
        plt.show(block=False)

        n_samples = 10
        plot_inputs = np.array(list(np.linspace(- 8, 8, num=200)) + [0., 1.])
        plot_inputs.sort()
        ix_0 = np.where(plot_inputs == 0.)[0]
        ix_1 = np.where(plot_inputs == 1.)[0]

        f_bnn_gpp = sample_bnn(plot_inputs, n_samples, prior_params)  # f ~ p_bnn (f|phi)
        f_bnn = sample_bnn(plot_inputs, n_samples)  # f ~ p_bnn (f)
        f_gp = sample_gpp(plot_inputs, n_samples)  # f ~ p_gp  (f)

        # Plot samples of functions from the bnn and gp priors.
        for axes in ax: axes.cla()
        ax[0].plot(plot_inputs, f_gp, color='green')
        ax[1].plot(plot_inputs, f_bnn_gpp, color='red')
        ax[2].plot(plot_inputs, f_bnn, color='blue')
        for i in range(n_samples):
            ax[0].scatter([0., 1.], [f_gp[ix_0, i], f_gp[ix_1, i]], color='green', marker='o')
            ax[1].scatter([0., 1.], [f_bnn_gpp[ix_0, i], f_bnn_gpp[ix_1, i]], marker='o')
            ax[2].scatter([0., 1.], [f_bnn[ix_0, i], f_bnn[ix_1, i]], color='blue', marker='o')
        plt.draw()
        plt.pause(1.0 / 40.0)
        plt.clf()


    def callback_kl(prior_params, iter, g):
        # print(iter)
        # Plot samples of functions from the bnn and gp priors.
        if plot_during:
            # plot_lines(prior_params)
            plot_contours(prior_params)

        print("Iteration {} KL {} ".format(iter, kl(prior_params, iter)))

    # ----------------------------------------------------------------

    # Initialize the variational prior params HERE
    # the functions drawn from the optimized bnn prior are heavily
    # dependent on which initialization scheme used here

    init_prior_var_params = init_bnn_params(num_weights, scale=-1.5)

    # ---------------------- MINIMIZE THE KL --------------------------

    prior_params = adam(grad_kl, init_prior_var_params,
                        step_size=0.1, num_iters=iters_1, callback=callback_kl)
