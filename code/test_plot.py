import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)

plt.plot(t,s)
plt.title(r'$\alpha_i > \beta_i$')
plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$')
plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$')
plt.xlabel(r'\textbf{time}(s)')
plt.ylabel('volts (mV)')
plt.show()