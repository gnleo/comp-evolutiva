"""
Created on Tue 2020 27 Oct

@author: gnleo
"""

# library's --------
from ag_functions import *
import matplotlib.pyplot as plt

# variables ----------
# 6 casas decimais => 28 bits
tm = 0.01
tc = 0.75
bits = 56
split = int (bits / 2)
pop_size = 100
population = []
children = []
pop_children = []
average_fitness = []
bad_fitness = []
best_fitness = []
pop_fitness = np.zeros(pop_size)
children_fitness = np.zeros(pop_size)
generation = 1000
repetitions = 10

# main ----------
population = generate_population(pop_size, bits)

for i in range(pop_size):
    # divide cromosso para conversão
    x_bin = population[i][ : split ]
    y_bin = population[i][ split : ]
    # executa conversão binário para inteiro
    x_int = bin_2_int(x_bin, bits)
    y_int = bin_2_int(y_bin, bits)
    # preenche vetor Fitness da população
    pop_fitness[i] = fitness(x_int, y_int)
    # pop_fitness[i] = translate_fitness(x_int, y_int)

save('/02/1/12/generation_i', bits, population)


for k in range(repetitions):
    
    print('INÍCIO PROCESSO EVOLUTIVO {}'.format(k))

    for i in range(generation):
        while(int(len(pop_children)/bits) != pop_size):

            # executa somatório dos valores de fitness da população
            fitness_sum = sum_fitness(pop_fitness)

            # seleciona os índices de indivíduos aptos ao cruzamento
            index_parent_1 = roulette(pop_size, fitness_sum, pop_fitness)
            index_parent_2 = roulette(pop_size, fitness_sum, pop_fitness)

            children = crossover(population[index_parent_1], population[index_parent_2], bits, tc)

            # se houve cruzamento
            if children is not None:
                # executa procedimento de mutação
                children = mutation(children, tm)

                # adiciona filhos para nova população (geração)
                pop_children = np.append(pop_children, children)

        population = np.reshape(pop_children, (pop_size, bits))
        pop_children = []

        save('/02/1/12/loop_{}/generation_{}'.format(k, i), bits, population)

        # realiza novo cálculo de fitness
        for i in range(pop_size):
            # divide cromosso para conversão
            x_bin = population[i][ : split ]
            y_bin = population[i][ split : ]
            # executa conversão binário para inteiro
            x_int = bin_2_int(x_bin, bits)
            y_int = bin_2_int(y_bin, bits)
            # preenche vetor Fitness da população
            pop_fitness[i] = fitness(x_int, y_int)
            # pop_fitness[i] = translate_fitness(x_int, y_int)


        average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / pop_size))
        bad_fitness = np.append(bad_fitness, select_bad_fitness(pop_fitness))
        best_fitness = np.append(best_fitness, select_best_fitness(pop_fitness))

    print('FIM PROCESSO EVOLUTIVO {}'.format(k))

average_fitness = np.reshape(average_fitness, (repetitions, generation))
bad_fitness = np.reshape(bad_fitness, (repetitions, generation))
best_fitness = np.reshape(best_fitness, (repetitions, generation))

save('/02/1/12/average_fitness', generation, average_fitness)
save('/02/1/12/bad_fitness', generation, bad_fitness)
save('/02/1/12/best_fitness', generation, best_fitness)

for g in range(repetitions):
    # gera gráfico de resultados
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(best_fitness[g], label='melhor')  # ... and some more.
    ax.plot(average_fitness[g], label='médio')  # Plot some data on the axes.
    ax.plot(bad_fitness[g], label='pior')  # Plot more data on the axes...
    ax.set_xlabel('iteração')  # Add an x-label to the axes.
    ax.set_ylabel('fitness')  # Add a y-label to the axes.
    ax.set_title("Processo Evolutivo loop {}: {} iterações".format(g, generation))  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.show()