#oppgave 1 konvulusjon
import numpy as np
from numpy import pi, exp, sin
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, sin, log10
import matplotlib.pyplot as plt
from scipy.signal import tukey
import scipy.io

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



#oppgave 2a
h1 = [0.0002, 0.0001, -0.0001, -0.0005, -0.0011, -0.0017, -0.0019, 
    -0.0016, -0.0005, 0.0015, 0.0040, 0.0064, 0.0079, 0.0075, 0.0046, 
    -0.0009, -0.0084, -0.0164, -0.0227, -0.0248, -0.0203, -0.0079, 
    0.0127, 0.0400, 0.0712, 0.1021, 0.1284, 0.1461, 0.1523, 0.1461, 
    0.1284, 0.1021, 0.0712, 0.0400, 0.0127, -0.0079, -0.0203, -0.0248, 
    -0.0227, -0.0164, -0.0084, -0.0009, 0.0046, 0.0075, 0.0079, 0.0064, 
    0.0040, 0.0015, -0.0005, -0.0016, -0.0019, -0.0017, -0.0011, 
    -0.0005, -0.0001, 0.0001, 0.0002]

h2 = [-0.0002, -0.0001, 0.0003, 0.0005, -0.0001, -0.0009, -0.0007, 
    0.0007, 0.0018, 0.0005, -0.0021, -0.0027, 0.0004, 0.0042, 0.0031, 
    -0.0028, -0.0067, -0.0023, 0.0069, 0.0091, -0.0010, -0.0127, 
    -0.0100, 0.0077, 0.0198, 0.0075, -0.0193, -0.0272, 0.0014, 0.0386, 
    0.0338, -0.0246, -0.0771, -0.0384, 0.1128, 0.2929, 0.3734, 0.2929, 
    0.1128, -0.0384, -0.0771, -0.0246, 0.0338, 0.0386, 0.0014, -0.0272, 
    -0.0193, 0.0075, 0.0198, 0.0077, -0.0100, -0.0127, -0.0010, 0.0091, 
    0.0069, -0.0023, -0.0067, -0.0028, 0.0031, 0.0042, 0.0004, -0.0027, 
    -0.0021, 0.0005, 0.0018, 0.0007, -0.0007, -0.0009, -0.0001, 0.0005, 
    0.0003, -0.0001, -0.0002]
#importerer alle sub-filene
mat_fil = scipy.io.loadmat('vibishar.mat')

offset1 = np.array(mat_fil['offset1'])
offset2 = np.array(mat_fil['offset2'])
seis1 = np.array(mat_fil['seismogram1'])
seis2 = np.array(mat_fil['seismogram2'])
time = np.array(mat_fil['t']) 

print('lenth of time array:', len(time))
print('lenth of seis1 array:', len(seis1))
print('lenth of seis2 array:', len(seis2))
m,n = np.shape(seis1)
def filter_seis(seismogram, filter_type):
    m,n = np.shape(seismogram)
    filtered = np.zeros((m-1,n)) #tom array for filterert seismogram

    #filterer en rad om gangen ved å bruke hvor konvin funksjon:
    for i in range(9):  #første fem radene
        filtered[:,i] = konvin3190(seismogram[:,i],0, filter_type)
    return filtered

plt.title('filtret seismogram 1 med h1-filter ')
plt.plot(time[0:m-1], filter_seis(seis1, h1))
#plt.legend(['uten vindusfunksjon', 'med vindusfunksjon'])
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude av seismogram')
plt.show()

plt.title('filtret seismogram 1 med h2-filter ')
plt.plot(time[0:m-1], filter_seis(seis1, h2))
#plt.legend(['uten vindusfunksjon', 'med vindusfunksjon'])
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude av seismogram')
plt.show()
offset1_isolated = seis1[0:500, 0] #røde område
time_isolated = time[0:500]

#plotter trasen i over den bestemte tids-intervallet
plt.plot(time_isolated, offset1_isolated)
plt.title('trasen vi skal analysere (tids-domenet)')
plt.xlabel('tid [s]')
plt.ylabel('amplitude')
plt.show()

#plotter frekvens spekteret:
sampling_f = 1/(time[2]-time[1])
f1, fs = frekspekin3190(offset1_isolated, 200, sampling_f)
plt.title('frekvensspekter til bestemt trasen')
plt.plot(fs, 20*log10(abs(f1)))
plt.xlabel('frekvens [Hz]')
plt.ylabel('amplitude')
plt.show()
vin_of1 = tukey(len(offset1_isolated), 0.5)*offset1_isolated
F_wo1, fs_wo1 = frekspekin3190(vin_of1, 200, sampling_f)

plt.plot(fs_wo1, 20*log10(abs(F_wo1)), fs, 20*log10(abs(f1)))
plt.legend(['med vindus', 'uten vindus'])
plt.show()

plt.plot(fs_wo1, abs(F_wo1), fs, abs(f1))
plt.show()