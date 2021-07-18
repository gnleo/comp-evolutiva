""" @author: gnleo """

from ee_functions import *
import matplotlib.pyplot as plt

sum_diversity = 0

children = []
pop_aux = []
pop_parents = []
pop_childrens = []

average_fitness = []
bad_fitness = []
best_fitness = []
diversity = []

pop_parents = generate_population()
fitness_parents = estimate_fitness(pop_parents, POP_PARENTS)

POP_SIZE_AUX = POP_CHILDREN + POP_PARENTS

fitness_aux = np.zeros(POP_SIZE_AUX)
fitness_parents = np.zeros(POP_PARENTS)


for g in range(GENERATION):
    while(int(len(pop_childrens) / (BITS + 1)) != POP_CHILDREN):
        fitness_sum = sum_fitness(fitness_parents)

        # seleciona os índices de indivíduos aptos ao cruzamento
        index_parent_1 = roulette(fitness_sum, fitness_parents)
        index_parent_2 = roulette(fitness_sum, fitness_parents)

        # cruzamento
        children = media_arithmetic_crossover_real(pop_parents[index_parent_1], pop_parents[index_parent_2])
        
        # adiciona filhos para nova população (geração)
        pop_childrens = np.append(pop_childrens, children)

    pop_childrens = np.reshape(pop_childrens, (POP_CHILDREN, BITS+1))

    pop_aux = np.append(pop_aux, pop_parents)
    pop_aux = np.append(pop_aux, mi_alfa(pop_childrens))
    pop_aux = np.reshape(pop_aux, (POP_SIZE_AUX, BITS+1))

    fitness_aux = estimate_fitness(pop_aux, POP_SIZE_AUX)

    # seleciona os melhores indivíduos da população aux
    best_indexes_aux = select_best_indexes(fitness_aux, POP_PARENTS)

    # atualiza matriz de pais
    for k in range(len(best_indexes_aux)):
        pop_parents[k] = pop_aux[int(best_indexes_aux[k])]

    # zera variáveis
    pop_aux = []
    pop_childrens = []

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