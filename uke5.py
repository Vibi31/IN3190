
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos

N = 128
ax = np.arange(-N , N, 1)
step = np.linspace(0, 2*N-1, 2*N)
xodd = np.zeros(2*N)
xeven = np.zeros(2*N)

X = np.zeros(2*N)
for n in range(2*N):
    X[n] = 1 + ax[n]/N + cos(2*pi*(ax[n]/N+ pi/4) )
    xodd[n] = 0.5*(X[n] - X[255-n])
    xeven[n] = 0.5*(X[n]+ X[255-n])

plt.plot(ax, X, label = 'x')
plt.plot(ax, xodd, label = 'odd')
plt.plot(ax, xeven, label = 'even')
plt.show()