import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, pi

T0 = 1
r = [0.25, 0.5, 0.75, 1]
N = 100
t = np.linspace(-T0/2, T0/2, N)

def wtuk(t,r):
    if -T0/2 <= t < (-T0/2+T0/2*r):
        return 1/2*(1-np.cos((t+T0/2)*(2*np.pi/r*T0)))     
    elif (-T0/2+T0/2*r) <= t <= (T0/2-T0/2*r):
        return 1
    elif (T0/2-T0/2*r) < t <= (T0/2):
        return 1/2*(1-np.cos((t-T0/2)*(2*np.pi/r*T0)))

b = np.zeros(N)
for a in range(len(r)):
    r_ = r[a]
    for i in range(len(t)):
        ti = t[i] 
        b[i] = wtuk(ti,r_)   
    plt.plot(t, b)
plt.show()

b = np.zeros(N)
for a in range(len(r)):
    r_ = r[a]
    for i in range(len(t)):
        ti = t[i] 
        b[i] = wtuk(ti,r_)  
    x = np.linspace(-30,30,len(t))     
    plt.plot(x, abs(np.fft.fftshift(b)))

plt.show()