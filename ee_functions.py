from operator import itemgetter
import math as m
import numpy as np
import random as rd
import csv
import os
from statistics import pstdev, stdev


RO = 2 # indica cruzamento | numero de indivíduos para cruzamento
MI = 100 # pais
LAMBIDA = 200 # filhos
BITS = 5
TM = 0.01
TC = 0.75

POP_PARENTS = MI
POP_CHILDREN = LAMBIDA

GENERATION = 500

""" Calcula valor de aptidão """
def fitness(x, y):
    num = m.pow( ( m.sin( m.sqrt( ( (x*x) + (y*y) ) ) ) ) , 2) - 0.5
    den = m.pow( ( ( 1 + ( 0.001 * ( (x*x) + (y*y) )) ) ), 2)

    return (0.5 - (num / den))


""" 
Cria população de indivíduos -> representação real 

indivíduo => | x1 | x2 | x3 | x4 | x5 | dp |
"""
def generate_population():
    population = []
    for i in range(POP_PARENTS):
        cromosso = np.zeros(BITS + 1)
        for j in range(BITS + 1):
            if(j == BITS):
                cromosso[j] = stdev(cromosso[:5])
                # cromosso[j] = pstdev(cromosso[:5])
            else:
                cromosso[j] = rd.uniform(-100, 100)

        population.append(cromosso)
    
    return population


""" Realiza cálculo de aptidão dos indivíduos -> f6(modificada) """
def estimate_fitness(population, POP_SIZE):
    pop_fitness = np.zeros(POP_SIZE)
    for k in range(POP_SIZE):
        x1 = population[k][0]
        x2 = population[k][1]
        x3 = population[k][2]
        x4 = population[k][3]
        x5 = population[k][4]
        pop_fitness[k] = fitness(x1,x2) + fitness(x2,x3) + fitness(x3,x4) + fitness(x4,x5) + fitness(x5,x1)
    
    return pop_fitness


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


""" 
Executa cálculo de densidade de probabilidade 
-> média == 0 | número aletório | desvio padrão do indivíduo
"""
def density_function(standard_deviation):
    # num_rand = rd.random()
    num_rand = rd.randint(-100, 100)
    num_var = num_rand ** 2
    den_var = standard_deviation ** 2
    return ( 1 / (standard_deviation * m.sqrt(2 * m.pi)) ) * ( m.exp( - (num_var / (2 * den_var))) )
    

""" Seleciona indices dos melhores indivíduos da população """
def select_best_indexes(list_interable, num_values):
    search_space = list_interable.copy()
    indexes = []
    for i in range(num_values):
        indexes = np.append(indexes, np.argmax(search_space))
        search_space[np.argmax(search_space)] = 0

    return indexes


""" Seleciona indices dos piores indivíduos da população """
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

""" Executa procedimento de soma dos valores de aptidão """
def sum_fitness(fitness):
    v_sum = 0
    for i in range(len(fitness)):
        v_sum += fitness[i]
        # print('v_sum = {} | v_fit = {}'.format(v_sum, x[i]))
    return v_sum


def mi_alfa(population):
    
    pop_n = np.zeros((POP_PARENTS, BITS + 1))

    for i in range(POP_PARENTS):
        for j in range(BITS):
            # increment = density_function(population[i][5])
            increment = rd.gauss(0.0, population[i][5])
            pop_n[i][j] = population[i][j] + increment
        
        pop_n[i][5] = population[i][5]

    return pop_n