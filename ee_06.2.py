""" @author: gnleo """

from ee_functions import *
import matplotlib.pyplot as plt

pop_parents = []
children = []
pop_childrens = []
fitness_parents = []


pop_parents = generate_population()
fitness_parents = estimate_fitness(pop_parents, POP_PARENTS)

# for g in range(GENERATION):
while(int(len(pop_childrens) / (BITS + 1)) != POP_CHILDREN):
    fitness_sum = sum_fitness(fitness_parents)

    # seleciona os índices de indivíduos aptos ao cruzamento
    index_parent_1 = roulette(fitness_sum, fitness_parents)
    index_parent_2 = roulette(fitness_sum, fitness_parents)

    # cruzamento
    children = media_arithmetic_crossover_real(pop_parents[index_parent_1], pop_parents[index_parent_2])
    
    # adiciona filhos para nova população (geração)
    pop_childrens = np.append(pop_childrens, children)

pop_childrens = np.reshape(pop_childrens, (POP_CHILDREN, BITS+1))
print("FIM PROCESSO EVOLUTIVO")