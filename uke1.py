import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, pi

#calculate functions
t = np.linspace(-1, 2.5, 1000)
f1 = cos(2*pi*t)
f2 = cos(2*pi*t + pi)
f3 = cos(8*pi*t)
f4 = cos(4*pi*t - pi/3)

#plot functions
plt.plot(t, f1)
plt.plot(t, f2)
plt.plot(t, f3)
plt.plot(t, f4)

#plot labels and name graphs
plt.legend(['f1', 'f2', 'f3', 'f4'])
plt.ylabel('f(t)')
plt.xlabel('t')
plt.show()


