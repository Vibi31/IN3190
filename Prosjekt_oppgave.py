#oppgave 1 konvulusjon
import numpy as np
import matplotlib.pyplot as plt

def konvin3190(x,ylen,h):
    M, N = len(h), len(x)
    y = np.zeros(M+N-1)
    
    for m in range(1, M):
        for n in range(1,N):
            o = n+m-1
            y(o) = y(o) + x(n)*h(m)
            
    if ylen == 0: #needs to return with the length of 'x'
        a, b = np.floor(0.5*(M-1)), np.ceil(0.5*(M-1) +1)
        return y[b:len(y)-a]
    else:
        return y
