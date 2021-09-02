import numpy as np
#oppgave 1a
x1 = np.array([-3, 2, 2, 1, 0, 4, -1])
x2 = np.array([-1, 2, -3, -3, 0, 0,0])
y1 =  x1 + x2
print('y1= ', y1)
#in terminal: y1 =[-4  4 -1 -2  0  4 -1]

#oppgave 1b og 1c
y2 = (1/3)*x1 + (2/3)*x2    #1b
y3 = x1*x2                  #1c

print('y2= ', y2)
#terminal: y2=  [-1.66, 2, -1.33, -1.66,  0, 1.33, -0.33]
print('y3= ', y3)
#terminal: y3=  [ 3  4 -6 -3  0  0  0]


