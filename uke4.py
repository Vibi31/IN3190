import matplotlib.pyplot as plt
import numpy as np

#pole limits:
R1 = -1/2       #I = imaginary, R = real
R2 = 1/2
I1 = -1/2
I2 = 1/2

Zcircle = plt.Circle((0,0),1, color = 'k', fill = True, clip_on = False)

plt.axis("equal")
"""
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
"""
plt.xlim(left=R1*2)
plt.xlim(right=R2*2)
plt.ylim(bottom=I1*2)
plt.ylim(top=I2*2)
plt.axhline(linewidth=2, color='k')
plt.axvline(linewidth=2, color='k')

##plt.grid(True)
plt.grid(color='k', linestyle='-.', linewidth=0.5)
plt.show()