""" @author: gnleo """

import os
import csv
import math as m
import random as rd
from fitness import *
from operator import itemgetter

# 6 casas decimais => 28 bits
TM = 0.01
TC = 0.75
BITS = 56
SPLIT = int(BITS / 2)
POP_SIZE = 100
GENERATION = 500
PERCENT = 10

REPETITIONS = 1

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


""" Executa procedimento de cruzamento um ponto de corte """
def crossover_one_point(parent_1, parent_2):
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


""" Executa cruzamento binário dois pontos de corte"""
def crossover_two_point(parent_1, parent_2):
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
            pop_fitness[i] = fitness_schaffer_4(x_int, y_int)
        else:
            pop_fitness[i] = fitness_displacement_schaffer_6(x_int, y_int)

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




""" Retorna os pontos de determinada população de indivíduos """
def map_points(path, type_struct):
    data = np.genfromtxt(os.getcwd() + path, delimiter=',')

    x = []
    y = []
    z = []

    for i in range(len(data[ 1 : ])):
        if(type_struct == 'R'):
            # representaçao real ------
            x = np.append(x, data[ i ][ 0 ])
            y = np.append(y, data[ i ][ 1 ])
        else:
            # representaçao binária ------
            x = np.append(x, round(bin_2_int(data[ i ][ : SPLIT ]), 6))
            y = np.append(y, round(bin_2_int(data[ i ][ SPLIT : ]), 6))

    for i in range(len(data[ 1 : ])):
        if(type_struct == 'R'):
            # representaçao real ------
            z = np.append(z, fitness_ackley(x[i],y[i]))
        else:
            # representaçao binária ------
            z = np.append(z, round(fitness_schaffer_4(x[i],y[i]), 6))

    return x, y, z