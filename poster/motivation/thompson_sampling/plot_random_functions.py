import numpy as np
import benchfunk
import reggie
import mwhutils.plotting as mp

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders

np.random.seed(1)


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    # XXX: replace this with a different function, perhaps benchfunk.Sinusoidal
    # (easy). the parameter is the noise of the blackbox.

    f = benchfunk.Matlab_example()
    bounds = f.bounds

    # get initial data and some test points.
    # XXX: ramp up the number of initial points if you want to ignore possibly
    # initial bad behavior without hyperparameter sampling.
    X = inits.init_latin(bounds, 3)
    Y = np.array([f(x_) for x_ in X])
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

    # initialize the model
    model = reggie.make_gp(0.01, 1.9, 0.1, 0)

    X = np.transpose(np.array(np.loadtxt("results_matlab/Xsamples4.txt"), ndmin = 2))
    Y = np.loadtxt("results_matlab/Ysamples4.txt")
    model.add_data(X, Y)

    # set a prior on the parameters
    # XXX: comment this block out to get it without hyper-parameter
    # marginalization.
#    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)
#    model.params['kern.rho'].set_prior('lognormal', 0, 100)
#    model.params['kern.ell'].set_prior('lognormal', 0, 10)
#    model.params['mean.bias'].set_prior('normal', 0, 20)
#    model = reggie.MCMC(model, n=20, rng=None)

    # create a new figure

    fig = mp.figure(rows=1, figsize=(6, 4))

    fig.clear()
    fig.hold()
    sample = []
    for i in range(10):
        sample.append(model.sample(np.expand_dims(x, 1)))
        fig.plot(x, sample[ -1 ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('01_samples.pdf')

    fig = mp.figure(rows=1, figsize=(6, 4))

    i = 0
    fig.clear()
    fig.hold()
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('01_samples_selected_sample.pdf')

    fig.clear()
    fig.hold()
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], sample[ i ][ sample[ i ].argmax() ], label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.vline(x[ sample[ i ].argmax() ], color = "red", ymax = 1.0 * (sample[ i ][ sample[ i ].argmax() ] + 7) / np.abs(4 + 7))
    fig.remove_ticks(yticks=True)
    fig.save('01_samples_sample_and_maximum.pdf')

    i = 1
    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1, color = color)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i -1 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('02_samples_selected_sample.pdf')

    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], sample[ i ][ sample[ i ].argmax() ], label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i -1 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.vline(x[ sample[ i ].argmax() ], color = "red", ymax = 1.0 * (sample[ i ][ sample[ i ].argmax() ] + 7) / np.abs(4 + 7))
    fig.remove_ticks(yticks=True)
    fig.save('02_samples_sample_and_maximum.pdf')

    i = 2
    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('03_samples_selected_sample.pdf')

    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], sample[ i ][ sample[ i ].argmax() ], label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.vline(x[ sample[ i ].argmax() ], color = "red", ymax = 1.0 * (sample[ i ][ sample[ i ].argmax() ] + 7) / np.abs(4 + 7))
    fig.remove_ticks(yticks=True)
    fig.save('03_samples_sample_and_maximum.pdf')

    i = 3
    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('04_samples_selected_sample.pdf')

    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], sample[ i ][ sample[ i ].argmax() ], label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.vline(x[ sample[ i ].argmax() ], color = "red", ymax = 1.0 * (sample[ i ][ sample[ i ].argmax() ] + 7) / np.abs(4 + 7))
    fig.remove_ticks(yticks=True)
    fig.save('04_samples_sample_and_maximum.pdf')

    i = 4
    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 4 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('05_samples_selected_sample.pdf')

    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], sample[ i ][ sample[ i ].argmax() ], label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 4 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.vline(x[ sample[ i ].argmax() ], color = "red", ymax = 1.0 * (sample[ i ][ sample[ i ].argmax() ] + 7) / np.abs(4 + 7))
    fig.remove_ticks(yticks=True)
    fig.save('05_samples_sample_and_maximum.pdf')

    i = 5
    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 4 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 5 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('06_samples_selected_sample.pdf')

    fig.clear()
    fig.hold()
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    color = next(fig._lcolors)
    fig.plot(x, sample[ i ], zorder = 1)
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], sample[ i ][ sample[ i ].argmax() ], label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 4 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 5 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.vline(x[ sample[ i ].argmax() ], color = "red", ymax = 1.0 * (sample[ i ][ sample[ i ].argmax() ] + 7) / np.abs(4 + 7))
    fig.remove_ticks(yticks=True)
    fig.save('06_samples_sample_and_maximum.pdf')

    fig.clear()
    fig.hold()
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.scatter(x[ sample[ i ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 1].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 2 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 3 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 4 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.scatter(x[ sample[ i - 5 ].argmax() ], -7 + 0.2, label = "", zorder = 2, s = 80, marker = 'o', color = 'red')
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('final.pdf')

    fig.clear()
    fig.hold()
    fig.scatter(model.data[0].ravel(), model.data[1], label = "objective", zorder = 2, s = 80)
    fig.set_lim(ymin=-7, ymax=4, xmin = -1e-2, xmax = 1 + 1e-2)
    fig.remove_ticks(yticks=True)
    fig.save('initial.pdf')

