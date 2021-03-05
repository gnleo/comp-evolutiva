""" @author: gnleo """

from ee_functions import *
import matplotlib.pyplot as plt

sum_diversity = 0

pop_aux = []
population = []
pop_children = []
best_indexes_aux = []

fitness_aux = np.zeros(300)
fitness_parents = np.zeros(POP_PARENTS)

average_fitness = []
bad_fitness = []
best_fitness = []
diversity = []

POP_AUX = POP_CHILDREN + POP_PARENTS


# main

pop_parents = generate_population()

for g in range(GENERATION):
    # cria filhos a partir dos pais 
    pop_children = np.append(pop_children, mi_alfa(pop_parents))
    pop_children = np.append(pop_children, mi_alfa(pop_parents))
    # pop_children = np.reshape(pop_children, (POP_CHILDREN, BITS + 1))

    # concatena as matrizes (PAIS e FILHOS)
    pop_aux = np.append(pop_aux, pop_parents)
    pop_aux = np.append(pop_aux, pop_children)
    
    # refatora matriz auxiliar
    pop_aux = np.reshape(pop_aux, (POP_AUX, BITS + 1))

    fitness_aux = estimate_fitness(pop_aux, POP_AUX)

    # seleciona os melhores indivíduos da população aux
    best_indexes_aux = select_best_indexes(fitness_aux, POP_PARENTS)

    # atualiza matriz de pais
    for k in range(len(best_indexes_aux)):
        pop_parents[k] = pop_aux[int(best_indexes_aux[k])]

    # zera variáveis
    pop_aux = []
    pop_children = []

    # executa avaliação fitness dos indivíduos da geração atual
    fitness_parents = estimate_fitness(pop_parents, POP_PARENTS)

    value_diversity_generation = measure_diversity_distance_euclidean(pop_parents)
    sum_diversity = (sum_diversity + value_diversity_generation) / POP_PARENTS

    # calcula média fitness da geração atual
    average_fitness = np.append(average_fitness, ((sum_fitness(fitness_parents)) / POP_PARENTS))
    # seleciona melhor indivíduo da geração atual
    best_fitness = np.append(best_fitness, select_best_fitness(fitness_parents))
    # seleciona piores indivíduos da geração atual
    bad_fitness = np.append(bad_fitness, select_bad_fitness(fitness_parents))
    # diversidade
    diversity = np.append(diversity, sum_diversity)


# average_fitness = np.reshape(average_fitness, (1, GENERATION))


print('FIM PROCESSO EVOLUTIVO')

fig, (ax,bx) = plt.subplots(1,2) 

ax.plot(best_fitness, label='melhor')
ax.plot(average_fitness, label='média')
ax.plot(bad_fitness, label='pior')
bx.plot(diversity, label='diversidade')
# configuração legenda
ax.set_xlabel('iteração')
ax.set_ylabel('fitness')
# ax.set_title("Processo evolutivo - loop {}: {} iterações".format(g, GENERATION))
ax.legend() 

# salva gráfico em diretório específico
# plt.savefig(os.getcwd() + PATH_SAVE + '/evolution_{}.png'.format(g))
# exibe imagem
plt.show()