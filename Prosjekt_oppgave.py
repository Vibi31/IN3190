#oppgave 1 konvulusjon
import numpy as np
from numpy import pi, exp, sin
import matplotlib.pyplot as plt

#oppgave 1a konvulusjon 
def konvin3190(x,ylen,h):
    M, N = len(h), len(x)
    y = np.zeros(M+N-1)
    
    for m in range(1, M):
        for n in range(1,N):
            o = n+m-1
            y[o] = y[o] + x[n]*h[m]
            
    if ylen == 0: #needs to return with the length of 'x'
        a, b = int(np.floor(0.5*(M-1))), int(np.ceil(0.5*(M-1) +1))
        return y[b:int(M+N-1-a)]
    else:
        return y

#oppgave 1b frekvens
def frekspekin3190(x, N, fs):
    X = np.zeros(N)
    w = np.linspace(0, pi, N)    #N-data punkter som omega(w)
    for i in range(N):
        for j in range(len(x)):
            X[i] = X[i] +x[j]*exp(-1J*w[i]*(j-1))
    f = (w*fs)/(2*pi)

    return X,f

def filter_h(size):
    return (1/5)*np.ones(size)

#oppgave 1c plott 
f1, f2 = 10, 20                 #Hz
fs = 100                        #Hz
t_len = 5                       #sekunder 
t = np.linspace(0, t_len, fs)   #tidsarray
h = filter_h(4)                 #filter
x = sin(2*pi*f1*t) + sin(2*pi*f2*t)   #x[n] signal

y_sig = konvin3190(x,0, h)
N = 100
X, f_1 = frekspekin3190(x, N, fs)
Y, f_2 = frekspekin3190(y_sig, N, fs)

plt.plot(t, abs(X))
plt.show()

plt.plot(f_1, abs(X))
plt.show()