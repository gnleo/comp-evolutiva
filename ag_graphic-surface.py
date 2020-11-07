from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os

from ag_functions import fitness, bin_2_int

data = np.genfromtxt(os.getcwd() + '/bckp/ind_13.csv', delimiter=',')

# x = []
# y = []
# z = []

# for i in range(6):
#     x = np.append(x, round(bin_2_int(data[ i + 1 ][ : 14 ]), 4))
#     y = np.append(y, round(bin_2_int(data[ i + 1 ][ 14 : ]), 4))

# for i in range(6):
#     z = np.append(z, round(fitness(x[i],y[i]), 4))


x = round(bin_2_int(data[ 1 ][ : 14 ]), 4)
y = round(bin_2_int(data[ 1 ][ 14 : ]), 4)
z = round(fitness(x,y), 4)


fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = plt.axes(projection='3d')

# Make data.
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

f6 = 0.5 - (((Z)**2 - 0.5) / ( 1 + 0.001 * (X**2 + Y**2))**2)

# Plot the surface.
# surf = plt.plot([-10.75,-10.2],[-0.98,05.64],[1,0.98], 'ko')
# surf = plt.plot(x,y,z, 'ko')
surf = plt.plot([x],[y],[z], 'ko')
surf = ax.plot_surface(X, Y, f6, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
