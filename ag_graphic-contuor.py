from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
import os

from ag_functions import fitness, bin_2_int

data = np.genfromtxt(os.getcwd() + '/bckp/ind_14.csv', delimiter=',')

# x = []
# y = []
# z = []

# for i in range(6):
# x = np.append(x, round(bin_2_int(data[ 1 ][ : 14 ]), 4))
# y = np.append(y, round(bin_2_int(data[ 1 ][ 14 : ]), 4))

# # for i in range(6):
# z = np.append(z, round(fitness(x[i],y[i]), 4))
x = round(bin_2_int(data[ 1 ][ : 14 ]), 4)
y = round(bin_2_int(data[ 1 ][ 14 : ]), 4)
z = round(fitness(x,y), 4)

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)
X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

f6 = 0.5 - (((Z)**2 - 0.5) / ( 1 + 0.001 * (X**2 + Y**2))**2)

# ax.plot_surface(X, Y, f6, rstride=8, cstride=8, alpha=0.3)
# cset = plt.plot([-0.59, 0.8], [10.54, 9.3],[0.68, 0.7], 'ko')
cset = plt.plot([x],[y],[z], 'ko')
# cset = plt.plot(x,y,z, 'ko')
cset = ax.contour(X, Y, f6, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, f6, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, f6, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-13, 13)
ax.set_ylabel('Y')
ax.set_ylim(-13, 13)
ax.set_zlabel('Z')
ax.set_zlim(0, 1)

plt.show()
