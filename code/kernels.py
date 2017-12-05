import autograd.numpy as np
import autograd.numpy.random as npr
rs = npr.RandomState(0)

# will add a bunch of kernels and stuff


def L2_norm(x, xp):
    d = x[:, None] - xp[None, :]
    matrix_l2_norm = np.sum(d**2, axis=2)
    return matrix_l2_norm


def L1_norm(x, xp):
    d = x[:, None] - xp[None, :]
    d **= 2
    matrix_l1_norm = np.sum(d**0.5, axis=2)
    return matrix_l1_norm


def kernel(x, xp, param, f=np.exp, norm='l2'):
    """
    generalized covariance function
    os,ls are output and length scales respectively
    f is some differentiable function used to construct the kernel
    norm constructs a matrix of ||x-x'||^2_2 or ||x-x'||^2_1
    """
    os, ls = f(param[0]), f(param[1:])
    norm = L2_norm if norm == "l2" else L1_norm
    K = os * f(norm(x, xp)/ls)
    return K


def covariance(x, xp, kernel_params=0.1 * rs.randn(2)):
    output_scale = np.exp(kernel_params[0])
    length_scales = np.exp(kernel_params[1:])
    diffs = (x[:, None] - xp[None, :]) / length_scales
    cov = output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))
    return cov


def kernel_rbf(x, xp):
    s, l = 1.0, 1.0
    d = L2_norm(x, xp)
    k = s*np.exp(-0.5 * d/l**2)
    return k


def kernel_per(x, xp):
    s, p, l = 1.0, 3.0, 1.0
    d = L1_norm(x, xp)/p
    k = s*np.exp(-2 * (np.sin(np.pi*d)/l)**2)
    return k


def kernel_rq(x, xp, alpha=3):
    d = L2_norm(x, xp)
    k = 1/(1 + 0.5 * d/alpha)**alpha
    return k


def kernel_per_rbf(x, xp):
    return kernel_per(x, xp)*kernel_rbf(x, xp)


def kernel_lin(x, xp):
    c, s, h = 0, 1.0, 0
    x=x.ravel(); xp=xp.ravel()
    k = s*(x[:, None]-c)*(xp[None, :]-c)
    print(k.shape)
    return k


def kernel_lin_per(x, xp):
    return kernel_lin(x, xp)*kernel_per(x, xp)

kernel_dict = {"rbf": kernel_rbf,
               "per": kernel_per,
               "rq": kernel_rq,
               "lin": kernel_lin,
               "per-rbf": kernel_per_rbf,
               "lin-per": kernel_lin_per,
               }
