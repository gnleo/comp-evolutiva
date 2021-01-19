"""
Created on Tue 2020 27 Oct

@author: gnleo
"""

from operator import itemgetter
import math as m
import numpy as np
import random as rd
import csv
import os

# 6 casas decimais => 28 bits
TM = 0.01
TC = 0.75
BITS = 2
SPLIT = int(BITS / 2)
POP_SIZE = 100
GENERATION = 500
PERCENT = 1

REPETITIONS = 3

""" Calcula valor de precisão em casas decimais """
def precision_2_number_bits(value_precision, i_I = -100, i_F = 100):
    num = ((i_F - (i_I)) * (m.pow(10, value_precision)))
    return m.ceil( m.log(num) / m.log(2) )


""" Cria população de indivíduos -> representação binária """
def generate_population():
    population = []
    for i in range(POP_SIZE):
        cromosso = np.zeros(BITS)

        # alelo
        for j in range(BITS):
            cromosso[j] = rd.randint(0,1)

        population.append(cromosso)
        
    return population


""" Transforma valor binário para inteiro """
def bin_2_int(value_bit):
    soma = 0
    index = len(value_bit)

    for i in range(len(value_bit)):
        soma = soma + value_bit[i] * m.pow(2, (index - 1))
        index -= 1

    soma = (soma *  ( 200 / (( 2**(BITS/2) ) - 1) ) ) - 100
    return soma


""" Calcula valor de aptidão """
def fitness(x, y):
    num = m.pow( ( m.sin( m.sqrt( ( (x*x) + (y*y) ) ) ) ) , 2) - 0.5
    den = m.pow( ( ( 1 + ( 0.001 * ( (x*x) + (y*y) )) ) ), 2)

    return (0.5 - (num / den))


""" Calcula valor de aptidão deslocado """
def displacement_fitness(x, y):
    num = m.pow( ( m.sin( m.sqrt( ( (x*x) + (y*y) ) ) ) ) , 2) - 0.5
    den = m.pow( ( ( 1 + ( 0.001 * ( (x*x) + (y*y) )) ) ), 2)

    return (999.5 - (num / den))


""" Executa procedimento de roleta """
def roulette(sum_fitness, fitness):
    i = 0
    aux = 0
    limit = rd.random() * sum_fitness

    while(i < POP_SIZE and aux < limit):
        aux += fitness[i]
        i += 1

    return (i - 1)


""" Executa procedimento de soma dos valores de aptidão """
def sum_fitness(fitness):
    v_sum = 0
    for i in range(len(fitness)):
        v_sum += fitness[i]
        # print('v_sum = {} | v_fit = {}'.format(v_sum, x[i]))
    return v_sum


""" Executa procedimento de mutação """
def mutation(children):
    for i in range(len(children)):
        for j in range(len(children[i])):
            if(rd.random() < TM):
                if(children[i][j] == 1):
                    children[i][j] = 0
                else:
                    children[i][j] = 1

    return children


""" Executa procedimento de cruzamento """
def crossover(parent_1, parent_2):
    children = []
    if(rd.random() < TC):
        # cria vetor de indices inteiros até [BITS - 1]
        indexs = np.arange(0, BITS, 1)
        # sorteia indice de corte
        index_sort = rd.sample(list(indexs), 1)

        new_1 = np.zeros(BITS)
        new_2 = np.zeros(BITS)

        # print('\TCruzamento')
        if(index_sort == 0 or index_sort == (BITS-1)):
            new_1[index_sort] = parent_1[index_sort]
            new_1[index_sort : ] = parent_2[index_sort :]

            new_2[index_sort] = parent_2[index_sort]
            new_2[index_sort : ] = parent_1[index_sort :]
        else:
            new_1[ : index_sort[0] ] = parent_1[ : index_sort[0] ]
            new_1[ index_sort[0] : ] = parent_2[ index_sort[0] : ]

            new_2[ : index_sort[0] ] = parent_2[ : index_sort[0] ]
            new_2[ index_sort[0] : ] = parent_1[ index_sort[0] : ]

        children = np.append(children, new_1)
        children = np.append(children, new_2)
    else:
        children = np.append(children, parent_1)
        children = np.append(children, parent_2)
    
    return np.reshape(children, (2,BITS))


""" Executa cruzamento binário """
def crossover_binary(parent_1, parent_2):
    children = []
    if(rd.random() < TC):

        new_1 = np.zeros(BITS)
        new_2 = np.zeros(BITS)

        # cria vetor de indices inteiros até [BITS - 1]
        indexs = np.arange(0, BITS, 1)
        # sorteia indice de corte
        boolean = True
        while(boolean == True):
            index_sort_1 = rd.sample(list(indexs), 1)
            index_sort_2 = rd.sample(list(indexs), 1)

            if(index_sort_1 < index_sort_2 and ((index_sort_2[0] - index_sort_1[0]) >= 2)):
                boolean = False

        # cruzamento
        if(index_sort_1[0] == 0 and index_sort_2[0] == (6-1)):
            # print('1')
            # print('index_1 = {}, index_2 = {}'.format(index_sort_1[0], index_sort_2[0]))
            index_plus = 1

            new_1[ index_sort_1[0] ] = parent_1[ index_sort_1[0] ]
            new_1[ index_plus : index_sort_2[0] ] = parent_2[ index_plus : index_sort_2[0] ]
            new_1[ index_sort_2[0] ] = parent_1[ index_sort_2[0] ]

            new_2[ index_sort_1[0] ] = parent_2[ index_sort_1[0] ]
            new_2[ index_plus : index_sort_2[0] ] = parent_1[ index_plus : index_sort_2[0] ]
            new_2[ index_sort_2[0] ] = parent_2[ index_sort_2[0] ]
        elif(index_sort_1[0] == 0):
            # print('2')
            # print('index_1 = {}, index_2 = {}'.format(index_sort_1[0], index_sort_2[0]))
            index_plus = 1

            new_1[ index_sort_1[0] ] = parent_1[ index_sort_1[0] ]
            new_1[ index_plus : index_sort_2[0] ] = parent_2[ index_plus : index_sort_2[0] ]
            new_1[ index_sort_2[0] : ] = parent_1[ index_sort_2[0] : ]

            new_2[ index_sort_1[0] ] = parent_2[ index_sort_1[0] ]
            new_2[ index_plus : index_sort_2[0] ] = parent_1[ index_plus : index_sort_2[0] ]
            new_2[ index_sort_2[0] : ] = parent_2[ index_sort_2[0] : ]
        else:
            # print('3')
            # print('index_1 = {}, index_2 = {}'.format(index_sort_1[0], index_sort_2[0]))
            new_1[ : index_sort_1[0] ] = parent_1[ : index_sort_1[0] ]
            new_1[ index_sort_1[0] : index_sort_2[0] ] = parent_2[ index_sort_1[0] : index_sort_2[0] ]
            new_1[ index_sort_2[0] : ] = parent_1[ index_sort_2[0] : ]

            new_2[ : index_sort_1[0] ] = parent_2[ : index_sort_1[0] ]
            new_2[ index_sort_1[0] : index_sort_2[0] ] = parent_1[ index_sort_1[0] : index_sort_2[0] ]
            new_2[ index_sort_2[0] : ] = parent_2[ index_sort_2[0] : ]

        children = np.append(children, new_1)
        children = np.append(children, new_2)

    else:
        children = np.append(children, parent_1)
        children = np.append(children, parent_2)
    
    return  np.reshape(children, (2,BITS))


""" Executa cruzamento binário uniforme """
def uniform_crossover_binary(parent_1, parent_2):
    children = []
    if(rd.random() < TC):
        mask = np.zeros(BITS)
        new_1 = np.zeros(BITS)
        new_2 = np.zeros(BITS)

        for i in range(BITS):
            mask[i] = rd.randint(0,1)
            if(mask[i] == 1):
                new_1[i] = parent_2[i]
                new_2[i] = parent_1[i]
            else:
                new_1[i] = parent_1[i]
                new_2[i] = parent_2[i]

        children = np.append(children, new_1)
        children = np.append(children, new_2)
    else:
        children = np.append(children, parent_1)
        children = np.append(children, parent_2)
    
    return  np.reshape(children, (2,BITS))


""" Salva estrutura da população """
def save(name, first_char_name_arq, structure):

    aux = name.split('/{}'.format(first_char_name_arq))
    folder = os.getcwd()  + aux[0]

    if not os.path.exists(folder):
        os.makedirs(folder)

    # cria vetor de indices inteiros até [BITS - 1]
    columns = np.arange(0, BITS, 1)
    # estabelece path de destino do arquivo
    complete_path = os.getcwd() + str(name) + '.csv'

    with open(complete_path, 'a+', newline = '') as arq:
        linha = csv.writer(arq, delimiter=',') # delimiter = '.'
        linha.writerow(columns)
        for element in structure:
            linha.writerow(element)


""" Seleciona os melhores indivíduos da população """
def select_best_indexes(list_interable, num_values):
    search_space = list_interable.copy()
    indexes = []
    for i in range(num_values):
        indexes = np.append(indexes, np.argmax(search_space))
        search_space[np.argmax(search_space)] = 0

    return indexes


""" Seleciona os piores indivíduos da população """
def select_worst_indexes(list_interable, num_values):
    search_space = list_interable.copy()
    indexes = []
    for i in range(num_values):
        indexes = np.append(indexes, np.argmin(search_space))
        search_space[np.argmin(search_space)] = 100000

    return indexes


"""" Seleciona melhor indivíduo da população """
def select_best_fitness(list_interable):
    return list_interable[np.argmax(list_interable)]


"""" Seleciona pior indivíduo da população """
def select_bad_fitness(list_interable):
    return list_interable[np.argmin(list_interable)]


""" Executa procedimento de elitismo """
def elitism(best_indexes, bad_indexes, pop, pop_children):
    aux = 0

    while(aux != len(bad_indexes)):
        # print(pop[int(round(bad_indexes[int(round(aux))]))])
        # print(pop_children[int(round(m_fit_f[int(round(aux))]))])
        pop_children[int(round(bad_indexes[int(round(aux))]))] = pop[int(round(best_indexes[int(round(aux))]))]
        aux += 1

    return pop_children


"""" Realiza cálculo de aptidão dos indivíduos -> representação binária """
def estimate_fitness(population, fitness_type='N'):
    pop_fitness = np.zeros(POP_SIZE)

    for i in range(POP_SIZE):
        # divide cromosso para conversão
        x_bin = population[i][ : SPLIT ]
        y_bin = population[i][ SPLIT : ]
        # executa conversão binário para inteiro
        x_int = bin_2_int(x_bin)
        y_int = bin_2_int(y_bin)
        # preenche vetor Fitness da população
        if(fitness_type == 'N'):
            pop_fitness[i] = fitness(x_int, y_int)
        else:
            pop_fitness[i] = displacement_fitness(x_int, y_int)

    return pop_fitness


""" Cria população de indivíduos -> representação real """
def generate_population_real():
    population = []
    for i in range(POP_SIZE):
        cromosso = np.zeros(BITS)
        for j in range(BITS):
            cromosso[j] = rd.uniform(-100, 100)

        population.append(cromosso)
    
    return population


"""" Realiza cálculo de aptidão dos indivíduos -> representação real """
def estimate_fitness_real(population):
    pop_fitness = np.zeros(POP_SIZE)
    for k in range(POP_SIZE):
        x = population[k][0]
        y = population[k][1]
        pop_fitness[k] = fitness(x,y)
    
    return pop_fitness


""" Executa procedimento de cruzamento -> representação real """
def crossover_real(parent_1, parent_2):
    children = []

    if(rd.random() < TC):

        new_1 = np.zeros(BITS)
        new_2 = np.zeros(BITS)

        new_1[0] = parent_1[0]
        new_1[1] = parent_2[1]

        new_2[0] = parent_2[0]
        new_2[1] = parent_1[1]

        children = np.append(children, new_1)
        children = np.append(children, new_2)

    else:
        children = np.append(children, parent_1)
        children = np.append(children, parent_2)
    
    return np.reshape(children, (2,BITS))


""" Executa procedimento de cruzamento com média aritmética -> representação real """
def media_arithmetic_crossover_real(parent_1, parent_2):
    children = []
    if(rd.random() < TC):
        new_1 = np.zeros(BITS)

        for i in range(BITS):
            new_1[i] = ((parent_1[i] + parent_2[i]) / 2)

        children = np.append(children, new_1)
    else:
        children = np.append(children, parent_1)
    
    return  np.reshape(children, (1,BITS))


""" Executa procedimento de mutação randomica uniforme -> representação real """
def uniform_random_mutation(children):
    for i in range(len(children)):
        for j in range(len(children[i])):
            if(rd.random() < TM):
                children[i][j] = rd.uniform(-100, 100)

    return children


""" Realiza cálculo de aptidão dos indivíduos -> f6(modificada) """
def estimate_fitness_real_f6_M(population):
    pop_fitness = np.zeros(POP_SIZE)
    for k in range(POP_SIZE):
        x1 = population[k][0]
        x2 = population[k][1]
        x3 = population[k][2]
        x4 = population[k][3]
        x5 = population[k][4]
        pop_fitness[k] = fitness(x1,x2) + fitness(x2,x3) + fitness(x3,x4) + fitness(x4,x5) + fitness(x5,x1)
    
    return pop_fitness


""" Medida de diversidade -> aptidão """
def measure_diversity_aptitude(media_fitness_population, best_fitness):
    return (media_fitness_population / best_fitness)


""" Medida de diversidade -> Hamming (melhor e pior indivíduos) """
def measure_diversity_hamming(population, pop_fitness):
    distance = 0
    best_index = select_best_indexes(pop_fitness, 1)
    bad_index = select_worst_indexes(pop_fitness, 1)

    for allele in range(BITS):
        if(population[int(round(best_index[0]))][allele] != population[int(round(bad_index[0]))][allele]):
            distance += 1

    return distance


""" Medida de diversidade -> Hamming (total) """
def measure_diversity_hamming_complete(population):
    distance = 0
    
    for i in range (POP_SIZE - 1):
        for j in range(1, POP_SIZE):
            for k in range(BITS):
                if(population[i][k] != population[j][k]):
                    distance += 1

    return distance


""" Medida de diversidade -> distância euclidiana """
def measure_diversity_distance_euclidean(population):
    distance = 0

    for i in range(POP_SIZE -1):
        for j in range(1, POP_SIZE):
            d = np.sqrt( (population[j][0] - population[i][0])**2 + (population[j][1] - population[i][1])**2 )
            distance += d

    return distance


# verificar ---------------------------------------

def linear_normalization(population, fitness, MIN, MAX, BITS, POP_SIZE):
    normalized_population = np.zeros([BITS, POP_SIZE])
    normalized_fitness = np.zeros(POP_SIZE)
    # cria matriz de indivíduos e seus respectivos valores de aptdão
    pop_complete = np.c_[population, fitness]

    # ordena os indivíduos de forma decrescente, de acordo com a última coluna
    aux = sorted(pop_complete, reverse=True, key=itemgetter(len(pop_complete)))

    for i in range(POP_SIZE):
        normalized_population[i] = aux[i][ : len(pop_complete)]
    
    value_n = MAX - MIN
    value_d = POP_SIZE - 1
    value = value_n / value_d

    # print('n = {} d = {} v = {}'.format(value_n, value_d, value))

    for j in range(POP_SIZE):
        normalized_fitness[j] = MIN + (value * ((j+1) - 1))

    return normalized_population, sorted(normalized_fitness, reverse=True)