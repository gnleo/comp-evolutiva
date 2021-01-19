""" @author: gnleo """

# library's --------
from ag_functions import *
import matplotlib.pyplot as plt

# variables ----------
sum_diversity = 0
population = []
children = []
pop_children = []
average_fitness = []
bad_fitness = []
best_fitness = []
diversity = []
pop_fitness = np.zeros(POP_SIZE)
children_fitness = np.zeros(POP_SIZE)

PATH_SAVE = "/05/4/aptidao"

# main ----------
population = generate_population_real()
POPULATION_COPY = population.copy()

save(PATH_SAVE + '/population_inicial', 'p', population)

pop_fitness = estimate_fitness_real(population)

# para executar o algoritmo N vezes -> alterar o valor da variável 'repetitions'
for k in range(REPETITIONS):
    
    print('INÍCIO PROCESSO EVOLUTIVO {}'.format(k))

    for i in range(GENERATION):
        
        # após o while a população de filhos é equivalente a população anterior
        # aqui 2 é o numero de bits para real
        while(int(len(pop_children)/BITS) != POP_SIZE):
            # executa somatório dos valores de fitness da população
            fitness_sum = sum_fitness(pop_fitness)

            # seleciona os índices de indivíduos aptos ao cruzamento
            index_parent_1 = roulette(fitness_sum, pop_fitness)
            index_parent_2 = roulette(fitness_sum, pop_fitness)

            children = media_arithmetic_crossover_real(population[index_parent_1], population[index_parent_2])
            
            # executa procedimento de mutação
            children = uniform_random_mutation(children)

            # adiciona filhos para nova população (geração)
            pop_children = np.append(pop_children, children)

        # realiza cálculo de fitness da população de filhos
        pop_children = np.reshape(pop_children, (POP_SIZE, BITS))
        children_fitness = estimate_fitness_real(pop_children)

        # ELITISMO
        # seleciona melhores indivíduos da população anterior 
        best_indexes = select_best_indexes(pop_fitness, PERCENT)
        # seleciona piores indivíduos da geração atual
        bad_indexes = select_worst_indexes(children_fitness, PERCENT)
        # executa elitismo -> altera os piores registros da população, pelos melhores registros do processo evolutivo
        population = elitism(best_indexes, bad_indexes, population, pop_children)

        save(PATH_SAVE + '/evolution_{}/population_{}'.format(k,i), 'p', pop_children)
        
        # zera população de filhos
        pop_children = []

        # calcula fitness da geração atual => pop_fitness 
        pop_fitness = estimate_fitness_real(population)

        # 1 -> 1.1
        # value_diversity_generation = measure_diversity_hamming_complete(population)

        # 1 -> 1.2 | 3 -> 3.2
        value_diversity_generation = measure_diversity_aptitude(((sum_fitness(pop_fitness)) / POP_SIZE), select_best_fitness(pop_fitness))
        
        # 3 -> 3.1
        # value_diversity_generation = measure_diversity_distance_euclidean(population)

        sum_diversity = (sum_diversity + value_diversity_generation) / POP_SIZE

        # executa preenchimento dos vetores de diversidade, média, pior e melhor (fitness)
        diversity = np.append(diversity, sum_diversity)
        average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / POP_SIZE))
        bad_fitness = np.append(bad_fitness, select_bad_fitness(pop_fitness))
        best_fitness = np.append(best_fitness, select_best_fitness(pop_fitness))

    print('FIM PROCESSO EVOLUTIVO {}'.format(k))

    population = POPULATION_COPY
    sum_diversity = 0

# executa controle para cada repetição do treinamento -> realizando um mapeamento matricial
diversity = np.reshape(diversity, (REPETITIONS, GENERATION))
average_fitness = np.reshape(average_fitness, (REPETITIONS, GENERATION))
bad_fitness = np.reshape(bad_fitness, (REPETITIONS, GENERATION))
best_fitness = np.reshape(best_fitness, (REPETITIONS, GENERATION))

# plotagem de gráfico
for g in range(REPETITIONS):
    # cria figura e box
    # ax1, ax2) = plt.subplots(1, 2)
    fig, (ax, bx) = plt.subplots(1,2) 

    # plota as curvas de desempenho
    ax.plot(best_fitness[g], label='melhor')
    ax.plot(average_fitness[g], label='médio')
    ax.plot(bad_fitness[g], label='pior')
    bx.plot(diversity[g], label='diversidade')
    # configuração legenda
    ax.set_xlabel('iteração')
    ax.set_ylabel('fitness')
    ax.set_title("Processo evolutivo - loop {}: {} iterações".format(g, GENERATION))
    bx.set_title("Medida de diversidade")
    ax.legend() 
    bx.legend() 

    # salva gráfico em diretório específico
    # plt.savefig(os.getcwd() + PATH_SAVE + '/evolution_{}.png'.format(g))
    # exibe imagem
    plt.show()