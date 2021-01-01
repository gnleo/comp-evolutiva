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


TM = 0.01
TC = 0.75
BITS = 5
POP_SIZE = 100

SPLIT = int(BITS / 2)


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

    return (0.5 - (num / den))

def displacement_fitness(x, y):
    num = m.pow( ( m.sin( m.sqrt( ( (x*x) + (y*y) ) ) ) ) , 2) - 0.5
    den = m.pow( ( ( 1 + ( 0.001 * ( (x*x) + (y*y) )) ) ), 2)
    return (999.5 - (num / den))

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



def save(name, first_char_name_arq, bits, structure):

    aux = name.split('/{}'.format(first_char_name_arq))
    folder = os.getcwd()  + aux[0]

    if not os.path.exists(folder):
        os.makedirs(folder)

    # cria vetor de indices inteiros até [bits - 1]
    columns = np.arange(0, bits, 1)
    # estabelece path de destino do arquivo
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


def generate_population_real():
    population = []
    for i in range(POP_SIZE):
        cromosso = np.zeros(BITS)
        for j in range(BITS):
            cromosso[j] = rd.uniform(-100, 100)

        population.append(cromosso)
    
    return population

def estimate_fitness_real(population):
    pop_fitness = np.zeros(POP_SIZE)
    for k in range(POP_SIZE):
        x = population[k][0]
        y = population[k][1]
        pop_fitness[k] = fitness(x,y)
    
    return pop_fitness


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

def uniform_random_mutation(children):
    for i in range(len(children)):
        for j in range(len(children[i])):
            if(rd.random() < TM):
                children[i][j] = rd.uniform(-100, 100)
                # break

    return children


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
