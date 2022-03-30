import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1.5, 1.5, 3)
y = np.linspace(-1.5, 1.5, 3)

X, Y = np.meshgrid(x, y)

Z = X+Y-1.5

print(X,"\n",Y,"\n",Z)

cs = plt.contourf(X, Y, Z, 0)

plt.show()