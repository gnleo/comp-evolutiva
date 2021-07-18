""" @author: gnleo """

from fitness import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from ag_functions import map_points


points_x = []
points_y = []
points_z = []

# points_x, points_y, points_z = map_points('/01/cruzamento_uniforme/population_inicial.csv', 'B')
points_x, points_y, points_z = map_points('/01/cruzamento_uniforme/population_499.csv', 'B')

# points_x, points_y, points_z = map_points('/04/cruzamento_media_aritmetica/population_inicial.csv', 'R')
# points_x, points_y, points_z = map_points('/04/cruzamento_media_aritmetica/evolution_0/population_499.csv', 'R')

fitness_min = points_z[np.argmin(points_z)]
fitness_max = points_z[np.argmax(points_z)]
index_min = np.argmin(points_z)
index_max = np.argmax(points_z)

# domínio da função
DELTA_S0 = -100
DELTA_SF = 100

X = np.arange(DELTA_S0, DELTA_SF, 0.25)
Y = np.arange(DELTA_S0, DELTA_SF, 0.25)
X, Y = np.meshgrid(X, Y)

# F = fitness_schaffer_6(X,Y)
# F = fitness_displacement_schaffer_6(X,Y)
F = fitness_schaffer_4(X,Y)
# F = fitness_one_max(X,Y)
# F = fitness_cross_in_tray(X,Y)
# F = fitness_ackley(X,Y)
# F = fitness_one_max(X,Y)



# estabelece os quadros a serem renderizados
fig, ax = plt.subplots(1,1)

# plota os pares ordenados (x,y) -> 2d
ax.plot(points_x,points_y,'ko')
# plota o contorno da função
ax.contour(X, Y, F, cmap=cm.coolwarm)

# estabelece limites x e y -> para visão de cima para baixo
ax.set_xlim(DELTA_S0, DELTA_SF)
ax.set_ylim(DELTA_S0, DELTA_SF)

plt.show()