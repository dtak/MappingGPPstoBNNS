
import numpy as np

np.random.seed(1)

def evaluate_y_noiseless(x):
    return 7 * np.sin(x)

def evaluate_y(x):
    return 7 * np.sin(x) + 3 * np.abs(np.cos(0.5 * x)) * np.random.randn(1)

start = -4
end = 4
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
y_ground_truth = np.zeros((size, 1))
for i in range(size):
    y_ground_truth[ i, 0 ] = evaluate_y_noiseless(x_ground_truth[ i, 0 ])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
axes = plt.gca()
axes.set_ylim([ -15, 15 ])
matplotlib.rcParams.update({'font.size': 15 })
plt.plot(x[ : n / 2 , 0 ], y[ : n / 2 , 0 ], 'o', label = 'Data point', markersize = 7)
plt.plot(x_ground_truth, y_ground_truth, 'r-', linewidth = 6, label = 'Ground truth')
plt.title('Training Data for Heteroskedastic Problem')
plt.legend(loc = 4)
plt.ylabel('y')
plt.xlabel('x')

plt.savefig('figure.pdf', bbox_inches = 'tight')
