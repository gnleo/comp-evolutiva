"""
Created on Tue 2020 27 Oct

@author: gnleo
"""

# library's --------
from ag_functions import *

# variables ----------
# 6 casas decimais => 28 bits
tm = 0.10
tc = 0.75
bits = 28
split = int (bits / 2)
pop_size = 100
population = []
children = []
pop_children = []
average_fitness = []
pop_fitness = np.zeros(pop_size)
children_fitness = np.zeros(pop_size)
epoch = 1000

# main ----------
population = generate_population(pop_size, bits)

for i in range(pop_size):
    # divide cromosso para conversão
    x_bin = population[i][ : split ]
    y_bin = population[i][ split : ]
    # executa conversão binário para inteiro
    x_int = bin_2_int(x_bin)
    y_int = bin_2_int(y_bin)
    # preenche vetor Fitness da população
    pop_fitness[i] = fitness(x_int, y_int)

best_indexes = select_better_indexes(pop_fitness)

save('individuos_pop_I', bits, best_indexes, population)

print('Executando treinamento')
for i in range(epoch):
    # executa somatório dos valores de fitness da população
    fitness_sum = sum_fitness(pop_fitness)
    # seleciona os índices de indivíduos aptos ao cruzamento
    index_parent_1 = roulette(pop_size, fitness_sum, pop_fitness)
    index_parent_2 = roulette(pop_size, fitness_sum, pop_fitness)
    children = crossover(population[index_parent_1], population[index_parent_2], bits, tc)
    # se houve cruzamento
    if children is not None:
        # adiciona filhos para nova população (geração)
        pop_children = np.append(pop_children, children)
        if(int(len(pop_children)/bits) == pop_size):
            print('\tMódulo da população')
            pop_children = np.reshape(pop_children, (100, bits))
            # analisa os filhos gerados
            for j in range(len(pop_children)):
                pop_children[j] = mutation(pop_children[j], bits, tm)
                c_x_bin = pop_children[j][ : split ]
                c_y_bin = pop_children[j][ split : ]
                # executa conversão binário para inteiro
                x_int = bin_2_int(c_x_bin)
                y_int = bin_2_int(c_y_bin)
                # preenche vetor Fitness dos filhos
                children_fitness[j] = fitness(x_int, y_int)
                # executa competição entre nova geração e anterior
                if(pop_fitness[j] < children_fitness[j]):
                    # print('\tAlteração da população atual indice = {}'.format(j))
                    population[j] = pop_children[j]
                    pop_fitness[j] = children_fitness[j]

            best_indexes = select_better_indexes(pop_fitness)

            save(str(epoch), bits, best_indexes, population)
            # zera a matriz pop_children
            pop_children = []

    average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / pop_size))


import matplotlib.pyplot as plt
plt.title('Média fitness')
plt.plot(average_fitness)
plt.show()
