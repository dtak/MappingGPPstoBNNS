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