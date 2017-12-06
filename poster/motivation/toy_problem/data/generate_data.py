
import numpy as np

np.random.seed(1)

def evaluate_y1_noiseless(x):
    return 10 * np.sin(x)

def evaluate_y2_noiseless(x):
    return 10 * np.cos(x)

def evaluate_y(x):
    if np.random.randn(1) < 0:
        return evaluate_y1_noiseless(x) + np.random.randn(1)
    else:
        return evaluate_y2_noiseless(x) + np.random.randn(1)

start = -2
end = 2
n = 2000
x = np.random.uniform(low = start, high = end, size = n)[ : , None ]
y = np.zeros((n, 1))
for i in range(n):
    y[ i, 0 ] = evaluate_y(x[ i, 0 ])

np.savetxt('x_train.txt', x[ : n / 2 , 0 ])
np.savetxt('y_train.txt', y[ : n / 2 , 0 ])
np.savetxt('x_test.txt', x[ n / 2 : , 0 ])
np.savetxt('y_test.txt', y[ n / 2 : , 0 ])


size = 100
x_ground_truth = np.linspace(start, end, size)[ :, None ]
y1_ground_truth = np.zeros((size, 1))
y2_ground_truth = np.zeros((size, 1))
for i in range(size):
    y1_ground_truth[ i, 0 ] = evaluate_y1_noiseless(x_ground_truth[ i, 0 ])
    y2_ground_truth[ i, 0 ] = evaluate_y2_noiseless(x_ground_truth[ i, 0 ])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
axes = plt.gca()
axes.set_ylim([ -15, 15 ])
matplotlib.rcParams.update({'font.size': 15 })
plt.plot(x[ : n / 2 , 0 ], y[ : n / 2 , 0 ], 'o', label = 'Data point', markersize = 7)
plt.plot(x_ground_truth, y1_ground_truth, 'r-', linewidth = 6, label = 'Ground truth')
plt.plot(x_ground_truth, y2_ground_truth, 'r-', linewidth = 6)
plt.title('Training Data for Bi-modal Problem')
plt.legend(loc = 4)
plt.ylabel('y')
plt.xlabel('x')

plt.savefig('figure.pdf', bbox_inches = 'tight')
