"""
Created on Tue 2020 27 Oct

@author: gnleo
"""

# library's --------
from ag_functions import *
import matplotlib.pyplot as plt

# variables ----------
population = []
children = []
pop_children = []
average_fitness = []
bad_fitness = []
best_fitness = []
pop_fitness = np.zeros(POP_SIZE)
children_fitness = np.zeros(POP_SIZE)

PATH_SAVE = "/04/2_elitismo_singular/dois_pontos_corte"

# main ----------
population = generate_population()
POPULATION_COPY = population.copy()

# save(PATH_SAVE + '/population_inicial', 'p', population)

pop_fitness = estimate_fitness(population)

# para executar o algoritmo N vezes -> alterar o valor da variável 'repetitions'
for k in range(REPETITIONS):
    
    print('INÍCIO PROCESSO EVOLUTIVO {}'.format(k))

    for i in range(GENERATION):
        
        # após o while a população de filhos é equivalente a população anterior
        while(int(len(pop_children)/BITS) != POP_SIZE):
            # executa somatório dos valores de fitness da população
            # population, pop_fitness = linear_normalization(population, pop_fitness, 1, 20, BITS, POP_SIZE)

            fitness_sum = sum_fitness(pop_fitness)

            # seleciona os índices de indivíduos aptos ao cruzamento
            index_parent_1 = roulette(fitness_sum, pop_fitness)
            index_parent_2 = roulette(fitness_sum, pop_fitness)

            children = crossover_binary(population[index_parent_1], population[index_parent_2])
            
            # executa procedimento de mutação
            children = mutation(children)

            # adiciona filhos para nova população (geração)
            pop_children = np.append(pop_children, children)

        # realiza cálculo de fitness da população de filhos
        pop_children = np.reshape(pop_children, (POP_SIZE, BITS))
        children_fitness = estimate_fitness(pop_children)

        # ELITISMO
        # seleciona melhores indivíduos da população anterior 
        best_indexes = select_best_indexes(pop_fitness, percent)
        # seleciona piores indivíduos da geração atual
        bad_indexes = select_worst_indexes(children_fitness, percent)
        # executa elitismo -> altera os piores registros da população, pelos melhores registros do processo evolutivo
        population = elitism(best_indexes, bad_indexes, population, pop_children)
        
        # save(PATH_SAVE + '/evolution_{}/population_{}'.format(k,i), 'p', population)
        # zera população de filhos
        pop_children = []

        # calcula fitness da geração atual => pop_fitness 
        pop_fitness = estimate_fitness(population)

        # executa preenchimento dos vetores de média, pior e melhor (fitness)
        average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / POP_SIZE))
        bad_fitness = np.append(bad_fitness, select_bad_fitness(pop_fitness))
        best_fitness = np.append(best_fitness, select_best_fitness(pop_fitness))

    print('FIM PROCESSO EVOLUTIVO {}'.format(k))

    population = POPULATION_COPY

# executa controle para cada repetição do treinamento -> realizando um mapeamento matricial
average_fitness = np.reshape(average_fitness, (REPETITIONS, GENERATION))
bad_fitness = np.reshape(bad_fitness, (REPETITIONS, GENERATION))
best_fitness = np.reshape(best_fitness, (REPETITIONS, GENERATION))

# plotagem de gráfico
for g in range(REPETITIONS):
    # cria figura e box
    fig, ax = plt.subplots()  
    # plota as curvas de desempenho
    ax.plot(best_fitness[g], label='melhor')
    ax.plot(average_fitness[g], label='médio')
    ax.plot(bad_fitness[g], label='pior')
    # configuração legenda
    ax.set_xlabel('iteração')
    ax.set_ylabel('fitness')
    ax.set_title("Processo evolutivo - loop {}: {} iterações".format(g, GENERATION))  # Add a title to the axes.
    ax.legend() 
    # salva gráfico em diretório específico
    plt.savefig(os.getcwd() + PATH_SAVE + '/evolution_{}.png'.format(g))
    # exibe imagem
    plt.show()