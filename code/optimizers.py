import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten


def adam_minimax(grad_both, init_params_max, init_params_min, callback=None,
                 num_iters=100, max_iters=20,
                 step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam modified to do minimiax optimization, for instance to help with
    training generative adversarial networks."""

    x_max, unflatten_max = flatten(init_params_max)
    x_min, unflatten_min = flatten(init_params_min)

    m_max = np.zeros(len(x_max))
    v_max = np.zeros(len(x_max))
    m_min = np.zeros(len(x_min))
    v_min = np.zeros(len(x_min))
    for i in range(num_iters):
        g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                       unflatten_min(x_min), i)
        g_max, _ = flatten(g_max_uf)
        g_min, _ = flatten(g_min_uf)

        if callback: callback(unflatten_max(x_max), unflatten_min(x_min), i,
                              unflatten_max(g_max), unflatten_min(g_min))
        for i in range(max_iters):
            m_max = (1 - b1) * g_max      + b1 * m_max  # First  moment estimate.
            v_max = (1 - b2) * (g_max**2) + b2 * v_max  # Second moment estimate.
            mhat_max = m_max / (1 - b1**(i + 1))    # Bias correction.
            vhat_max = v_max / (1 - b2**(i + 1))
            x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)

        m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
        v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
        mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
        vhat_min = v_min / (1 - b2**(i + 1))
        x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)

    return unflatten_max(x_max), unflatten_min(x_min)

