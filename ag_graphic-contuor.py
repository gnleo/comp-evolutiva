"""
Created on Tue 2020 27 Oct

@author: gnleo
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
import os

from ag_functions import displacement_fitness, bin_2_int

data = np.genfromtxt(os.getcwd() + '/03/f6_N_BPS/training_0/population_9.csv.csv', delimiter=',')

bits = 56
split = int(bits/2)
x = []
y = []
z = []

for i in range(len(data[ 1 : ])):
    x = np.append(x, round(bin_2_int(data[ i ][ : split ], bits), 6))
    y = np.append(y, round(bin_2_int(data[ i ][ split : ], bits), 6))

for i in range(len(data[ 1 : ])):
    z = np.append(z, round(displacement_fitness(x[i],y[i]), 6))

z_max = z[np.argmax(z)]


X = np.arange(-100, 100, 0.5)
Y = np.arange(-100, 100, 0.5)
# X = np.arange(-10, 10, 0.5)
# Y = np.arange(-10, 10, 0.5)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

f6 = 0.5 - (((Z)**2 - 0.5) / ( 1 + 0.001 * (X**2 + Y**2))**2)


# fig = plt.figure()
# ax = fig.gca(projection='3d')

fig, ax = plt.subplots(1,1)
# cset = plt.plot([x],[y],[z], 'ko')
plt.plot(x,y,'ko')
# cset = plt.plot(x,y,'ko')
ax.contour(X, Y, f6, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, f6, cmap=cm.coolwarm)


ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

# ax.set_xlim(-13, 13)
# ax.set_ylim(-13, 13)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_zlim(0, 1000)
ax.set_title("Baixa PS - população 9")

plt.show()
