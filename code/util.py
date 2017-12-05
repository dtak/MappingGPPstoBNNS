import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.image
rs = npr.RandomState(0)

bell = lambda x: np.exp(-0.5*x**2)
cube =lambda x: 0.1*x*np.sin(x)


def build_toy_dataset(data="xsinx", n_data=70, noise_std=0.1):
    D = 1
    if data == "expx":
        inputs = np.linspace(0, 4, num=n_data)
        targets = bell(inputs) + rs.randn(n_data) * noise_std

    elif data == "cosx":
        inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                                  np.linspace(6, 8, num=n_data/2)])
        targets = np.cos(inputs) + rs.randn(n_data) * noise_std
        inputs = (inputs - 4.0) / 4.0

    elif data == "xsinx":
        inputs = np.linspace(0, 8, num=n_data)
        targets = cube(inputs) + rs.randn(n_data) * noise_std

    elif data == "lin":
        inputs = np.linspace(2, 6, num=n_data)
        targets = inputs+ rs.randn(n_data) * noise_std*14

    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets


def covariance(x, xp, kernel_params=0.1 * rs.randn(2)):
    output_scale = np.exp(kernel_params[0])
    length_scales = np.exp(kernel_params[1:])
    diffs = (x[:, None] - xp[None, :]) / length_scales
    cov = output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))
    return cov


def make_title(exp_num, x_num, samples, kern,
               arch, act, iters, scale, step):

    train_info = " exp_num {} x_num {} w samples {} " \
                 " kernel {} arch {} act_fn {} " \
                 "iters {} init_scale {} step_size {}".format(
                 exp_num, x_num, samples, kern, arch, act, iters, scale, step)

    return train_info