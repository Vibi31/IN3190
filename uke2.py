import numpy as np

#oppgave 1a
x1 = np.array([0, -3, 2, 2, 1, 0, 4, -1])
x2 = np.array([-1, 2, -3, -3, 0, 0, 0, 0])
y1 =  x1 + x2
print('y1= ', y1)
#in terminal: y1= [-1 -1 -1 -1  1  0  4 -1]

#oppgave 1b og 1c
y2 = (1/3)*x1 - (2/3)*x2    #1b
y3 = x1*x2                  #1c

print('y2= ', y2)
#terminal: y2= [0.66666667 -2.33333333  2.66666667  2.66666667  0.33333333  0.  1.33333333 -0.33333333]
print('y3= ', y3)
#terminal: y3= [ 0 -6 -6 -6  0  0  0  0]


#oppgave 4
def konvolver(a, b):
    conv = np.zeros(2*len(a))
    for x in range(2*len(a)):
        for y in range(2*len(b)):
            conv[x] = a[y] * b[y]


    
    if conv == np.convolve(a,b):
        print('success')
    else:
        print('something wrong')

    return conv
    
konvolver(x1, x2)