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
generation = 500
repetitions = 5
percent = 10

# main ----------
population = generate_population(pop_size, bits)
POPULATION_COPY = population.copy()

pop_fitness = estimate_fitness(population, pop_size, bits, split)

# para executar o algoritmo N vezes -> alterar o valor da variável 'repetitions'
for k in range(repetitions):
    
    print('INÍCIO PROCESSO EVOLUTIVO {}'.format(k))

    for i in range(generation):
        
        # após o while a população de filhos é equivalente a população anterior
        while(int(len(pop_children)/bits) != pop_size):
            # executa somatório dos valores de fitness da população
            fitness_sum = sum_fitness(pop_fitness)

            # seleciona os índices de indivíduos aptos ao cruzamento
            index_parent_1 = roulette(pop_size, fitness_sum, pop_fitness)
            index_parent_2 = roulette(pop_size, fitness_sum, pop_fitness)

            children = crossover(population[index_parent_1], population[index_parent_2], bits, tc)
            
            # executa procedimento de mutação
            children = mutation(children, tm)

            # adiciona filhos para nova população (geração)
            pop_children = np.append(pop_children, children)


        pop_children = np.reshape(pop_children, (pop_size, bits))
        
        # realiza novo cálculo de fitness
        children_fitness = estimate_fitness(pop_children, pop_size, bits, split)

        # seleciona melhores indivíduos da população anterior 
        best_indexes = select_best_indexes(pop_fitness, percent)
        # seleciona piores indivíduos da geração atual
        bad_indexes = select_worst_indexes(children_fitness, percent)
        # executa elitismo -> altera os piores registros da população, pelos melhores registros do processo evolutivo
        population = elitism(best_indexes, bad_indexes, population, pop_children)
        
        # zera população de filhos
        pop_children = []

        # calcula fitness da geração atual => pop_fitness 
        # -> pode ser comentado para reduzir a subida de aptidão dos piores indivíduos
        pop_fitness = estimate_fitness(population, pop_size, bits, split)

        # executa preenchimento dos vetores de média, pior e melhor (fitness)
        average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / pop_size))
        bad_fitness = np.append(bad_fitness, select_bad_fitness(pop_fitness))
        best_fitness = np.append(best_fitness, select_best_fitness(pop_fitness))

    print('FIM PROCESSO EVOLUTIVO {}'.format(k))
    population = POPULATION_COPY

# executa controle para cada repetição do treinamento -> realizando um mapeamento matricial
average_fitness = np.reshape(average_fitness, (repetitions, generation))
bad_fitness = np.reshape(bad_fitness, (repetitions, generation))
best_fitness = np.reshape(best_fitness, (repetitions, generation))

# plotagem de gráfico
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