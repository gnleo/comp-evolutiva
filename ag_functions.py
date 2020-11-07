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

def bin_2_int(value_bit):
    bits = 28
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

def mutation(children, bits, tm):
    if(rd.random() < tm):
        # print('\tMutation')
        # cria vetor de indices inteiros até [bits - 1]
        indexs = np.arange(0, bits, 1)
        # sorteia indice de corte
        index_sort = rd.sample(list(indexs), 1)

        # executa a mutação, inverte o valor do bit
        if (children[index_sort[0]] == 1):
            children[index_sort[0]] = 0
        else:
            children[index_sort[0]] = 1
        
        return children
    else:
        return children


def crossover(parent_1, parent_2, bits, tc):
    if(rd.random() < tc):
        # cria vetor de indices inteiros até [bits - 1]
        indexs = np.arange(0, bits, 1)
        # sorteia indice de corte
        index_sort = rd.sample(list(indexs), 1)
        if(index_sort[0] > 0 and index_sort[0] < bits):
            # print('\tCruzamento')
            new_1 = np.zeros(bits)
            new_2 = np.zeros(bits)

            # print('índice = {}'.format(index_sort))
            # print('pai_1 = {}'.format(parent_1))
            # print('pai_2 = {}'.format(parent_2))
            # print('splip_1 = {}'.format(parent_1[ : index_sort[0] ]))
            # print('splip_2 = {}'.format(parent_2[ index_sort[0] : ]))
            # print('splip_2.1 = {}'.format(parent_2[ : index_sort[0] ]))
            # print('splip_1.1 = {}'.format(parent_1[ index_sort[0] : ]))

            new_1[ : index_sort[0] ] = parent_1[ : index_sort[0] ]
            new_1[ index_sort[0] : ] = parent_2[ index_sort[0] : ]

            new_2[ : index_sort[0] ] = parent_2[ : index_sort[0] ]
            new_2[ index_sort[0] : ] = parent_1[ index_sort[0] : ]

            children = []
            children = np.append(children, new_1)
            children = np.append(children, new_2)

            # print('filho_1 = {}'.format(new_1))
            # print('filho_2 = {}'.format(new_2))

            children = np.reshape(children, (2,bits))

            return children
            
        else:
            # print('índice = {}'.format(index_sort))
            # print('indice não permite cruzamento')
            return None

    else:
        return None


def save(name, bits, indexes, structure):

    # cria vetor de indices inteiros até [bits - 1]
    columns = np.arange(0, bits, 1)

    complete_path = os.getcwd() + '/bckp/' + str(name) + '.csv'

    with open(complete_path, 'a+', newline = '') as arq:
        linha = csv.writer(arq, delimiter=',') # delimiter = '.'
        linha.writerow(columns)
        for element in indexes:
            linha.writerow(structure[int(element)])


def select_better_indexes(list_interable):
    search_space = list_interable.copy()
    indexes = []
    for i in range(6):
        indexes = np.append(indexes, np.argmax(search_space))
        search_space[np.argmax(search_space)] = 0

    return indexes