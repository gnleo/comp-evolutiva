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

def precision_2_number_bits(value_precision, i_I = -100, i_F = 100):
    num = ((i_F - (i_I)) * (m.pow(10, value_precision)))
    return m.ceil( m.log(num) / m.log(2) )

def generate_population(pop_size, bits):
    population = []
    for i in range(pop_size):
        cromosso = np.zeros(bits)

        # alelo
        for j in range(bits):
            cromosso[j] = rd.randint(0,1)

        population.append(cromosso)
        
    return population

def bin_2_int(value_bit, bits):
    soma = 0
    index = len(value_bit)

    for i in range(len(value_bit)):
        soma = soma + value_bit[i] * m.pow(2, (index - 1))
        index -= 1


    soma = (soma *  ( 200 / (( 2**(bits/2) ) - 1) ) ) - 100
    return soma

def fitness(x, y):
    num = m.pow( ( m.sin( m.sqrt( ( (x*x) + (y*y) ) ) ) ) , 2) - 0.5
    den = m.pow( ( ( 1 + ( 0.001 * ( (x*x) + (y*y) )) ) ), 2)

    return ((0.5 - num) / den)

def displacement_fitness(x, y):
    num = m.pow( ( m.sin( m.sqrt( ( (x*x) + (y*y) ) ) ) ) , 2) - 0.5
    den = m.pow( ( ( 1 + ( 0.001 * ( (x*x) + (y*y) )) ) ), 2)
    return ((999.5 - num) / den)    

def roulette(pop_size, sum_fitness, fitness):
    i = 0
    aux = 0
    limit = rd.random() * sum_fitness

    while(i < pop_size and aux < limit):
        aux += fitness[i]
        i += 1

    return (i - 1)

def sum_fitness(fitness):
    v_sum = 0
    for i in range(len(fitness)):
        v_sum += fitness[i]
        # print('v_sum = {} | v_fit = {}'.format(v_sum, x[i]))
    return v_sum

def mutation(children, tm):
    for i in range(len(children)):
        for j in range(len(children[i])):
            if(rd.random() < tm):
                if(children[i][j] == 1):
                    children[i][j] = 0
                else:
                    children[i][j] = 1

    return children

def crossover(parent_1, parent_2, bits, tc):
    children = []
    if(rd.random() < tc):
        # cria vetor de indices inteiros até [bits - 1]
        indexs = np.arange(0, bits, 1)
        # sorteia indice de corte
        index_sort = rd.sample(list(indexs), 1)

        new_1 = np.zeros(bits)
        new_2 = np.zeros(bits)

        # print('\tCruzamento')
        if(index_sort == 0 or index_sort == (bits-1)):
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
    
    return np.reshape(children, (2,bits))

def crossover_binary(parent_1, parent_2, bits, tc):
    children = []
    if(rd.random() < tc):

        new_1 = np.zeros(bits)
        new_2 = np.zeros(bits)

        # cria vetor de indices inteiros até [bits - 1]
        indexs = np.arange(0, bits, 1)
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
            index_plus = (index_sort_1[0] + 1)

            new_1[ index_sort_1[0] ] = parent_1[ index_sort_1[0] ]
            new_1[ index_plus : index_sort_2[0] ] = parent_2[ index_plus : index_sort_2[0] ]
            new_1[ index_sort_2[0] ] = parent_1[ index_sort_2[0] ]

            new_2[ index_sort_1[0] ] = parent_2[ index_sort_1[0] ]
            new_2[ index_plus : index_sort_2[0] ] = parent_1[ index_plus : index_sort_2[0] ]
            new_2[ index_sort_2[0] ] = parent_2[ index_sort_2[0] ]
        elif(index_sort_1[0] == 0):
            # print('2')
            # print('index_1 = {}, index_2 = {}'.format(index_sort_1[0], index_sort_2[0]))
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
    
    return  np.reshape(children, (2,bits))

def uniform_crossover_binary(parent_1, parent_2, bits, tc):
    children = []
    if(rd.random() < tc):
        mask = np.zeros(bits)
        new_1 = np.zeros(bits)
        new_2 = np.zeros(bits)

        for i in range(bits):
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
    
    return  np.reshape(children, (2,bits))



def save(name, bits, structure):

    # cria vetor de indices inteiros até [bits - 1]
    columns = np.arange(0, bits, 1)

    # complete_path = os.getcwd() + '/bckp/' + str(name) + '.csv'
    complete_path = os.getcwd() + str(name) + '.csv'

    with open(complete_path, 'a+', newline = '') as arq:
        linha = csv.writer(arq, delimiter=',') # delimiter = '.'
        linha.writerow(columns)
        for element in structure:
            linha.writerow(element)


def select_best_indexes(list_interable, num_values):
    search_space = list_interable.copy()
    indexes = []
    for i in range(num_values):
        indexes = np.append(indexes, np.argmax(search_space))
        search_space[np.argmax(search_space)] = 0

    return indexes

def select_worst_indexes(list_interable, num_values):
    search_space = list_interable.copy()
    indexes = []
    for i in range(num_values):
        indexes = np.append(indexes, np.argmin(search_space))
        search_space[np.argmin(search_space)] = 100000

    return indexes

def select_best_fitness(list_interable):
    return list_interable[np.argmax(list_interable)]

def select_bad_fitness(list_interable):
    return list_interable[np.argmin(list_interable)]


def elitism(best_indexes, bad_indexes, pop, pop_children):
    aux = 0

    while(aux != len(bad_indexes)):
        # print(pop[int(round(bad_indexes[int(round(aux))]))])
        # print(pop_children[int(round(m_fit_f[int(round(aux))]))])
        pop_children[int(round(bad_indexes[int(round(aux))]))] = pop[int(round(best_indexes[int(round(aux))]))]
        aux += 1

    return pop_children

def estimate_fitness(population, pop_size, bits, split, fitness_type='N'):
    pop_fitness = np.zeros(pop_size)

    for i in range(pop_size):
        # divide cromosso para conversão
        x_bin = population[i][ : split ]
        y_bin = population[i][ split : ]
        # executa conversão binário para inteiro
        x_int = bin_2_int(x_bin, bits)
        y_int = bin_2_int(y_bin, bits)
        # preenche vetor Fitness da população
        if(fitness_type == 'N'):
            pop_fitness[i] = fitness(x_int, y_int)
        else:
            pop_fitness[i] = displacement_fitness(x_int, y_int)

    return pop_fitness

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