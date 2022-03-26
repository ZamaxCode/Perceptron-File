import matplotlib.pyplot as plt
import numpy as np

# generate 101 x and y values between -10 and 10 
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

# make X and Y matrices representing x and y values of 2d plane
X, Y = np.meshgrid(x, y)

# compute z value of a point as a function of x and y (z = l2 distance form 0,0)
Z = X+Y-1.5

# plot filled contour map with 100 levels
cs = plt.contourf(X, Y, Z, 0)

# add default colorbar for the map
plt.colorbar(cs)
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()