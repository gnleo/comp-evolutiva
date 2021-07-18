""" @author: gnleo """

from fitness import *
import numpy as np
import random as rd

TM = 0.01
TC = 0.75
BITS = 2
POP_SIZE = 100
REPETITIONS = 1
GENERATION = 500

""" Especifica o domínio da função -> caracteriza o espaço de busca """
DELTA_S0 = -5
DELTA_SF = 5


""" Cria população de indivíduos -> representação real """
def generate_population_real():
    population = []
    for i in range(POP_SIZE):
        cromosso = np.zeros(BITS)
        for j in range(BITS):
            cromosso[j] = rd.uniform(DELTA_S0, DELTA_SF)

        population.append(cromosso)
    
    return population


"""" Realiza cálculo de aptidão dos indivíduos -> representação real """
def estimate_fitness_real(population):
    pop_fitness = np.zeros(POP_SIZE)
    for k in range(POP_SIZE):
        x = population[k][0]
        y = population[k][1]
        pop_fitness[k] = fitness_ackley(x,y)
    
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
                children[i][j] = rd.uniform(DELTA_S0, DELTA_SF)

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
        pop_fitness[k] = fitness_schaffer_6(x1,x2) + fitness_schaffer_6(x2,x3) + fitness_schaffer_6(x3,x4) + fitness_schaffer_6(x4,x5) + fitness_schaffer_6(x5,x1)
    
    return pop_fitness
# 