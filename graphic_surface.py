""" @author: gnleo """

from fitness import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ag_functions import map_points

# domínio da função
DELTA_S0 = -5
DELTA_SF = 5

X = np.arange(DELTA_S0, DELTA_SF, 0.25)
Y = np.arange(DELTA_S0, DELTA_SF, 0.25)
X, Y = np.meshgrid(X, Y)

# F = fitness_schaffer_6(X,Y)
# F = fitness_displacement_schaffer_6(X,Y)
# F = fitness_schaffer_4(X,Y)
# F = fitness_one_max(X,Y)
# F = fitness_cross_in_tray(X,Y)
F = fitness_ackley(X,Y)
# F = fitness_one_max(X,Y)


fig = plt.figure()

ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, F, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()