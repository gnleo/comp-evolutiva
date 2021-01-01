import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

num_loop = 4

PATH = os.getcwd()

DIRECTORY = '/04/1.2_cruzamento_uniforme'

PATH_READ_INIT = PATH + DIRECTORY + '/population_inicial.csv'
PATH_WRITE_INIT = PATH + DIRECTORY + '/map_begin_population.png'

PATH_READ_EVOLUTIONS = PATH + DIRECTORY + '/evolution_{}'.format(num_loop)

# populações a serem analisadas
populations = [ 0, 40, 60, 80, 99, \
                110, 140, 170, 180, 199, \
                209, 230, 250, 275, 299, \
                316, 330, 370, 380, 399, \
                430, 440, 460, 480, 499 ]

# True: gera mapa da população inicial | False: gera mapa das populações indicadas
# init = True
init = False

if(init):
    # importação da população
    data = np.genfromtxt(PATH_READ_INIT, delimiter=',')

    # adiciona matriz populacional
    ax = sns.heatmap(data[1 :], cmap=cm.binary)

    # configuração de legendas
    ax.set_title('População inicial 1.1')
    ax.set_xlabel('Cromossomos')
    ax.set_ylabel('Indivíduos')

    plt.savefig(PATH_WRITE_INIT)

    plt.show()
else:
    # percorre o índice das populações a serem analizadas
    for i in range(len(populations)):

        # importação da população
        data = np.genfromtxt(PATH_READ_EVOLUTIONS + '/population_{}.csv'.format(populations[i]), delimiter=',')

        # adiciona matriz populacional
        ax = sns.heatmap(data[1 :], cmap=cm.binary)

        # configuração de legendas
        ax.set_title('População iteração {}'.format(populations[i]))
        ax.set_xlabel('Cromossomos')
        ax.set_ylabel('Indivíduos')

        plt.savefig(PATH_READ_EVOLUTIONS + '/map_population_{}.png'.format(populations[i]))

        plt.show()
