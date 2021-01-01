"""
Created on Tue 2020 27 Oct

@author: gnleo
"""

# library's --------
from ag_functions import *
import matplotlib.pyplot as plt

# variables ----------
# 6 casas decimais => 28 bits
tm = 0.01
tc = 0.75
bits = 56
split = int (bits / 2)
pop_size = 100
population = []
children = []
pop_children = []
average_fitness = []
bad_fitness = []
best_fitness = []
pop_fitness = np.zeros(pop_size)
children_fitness = np.zeros(pop_size)
generation = 500
percent = 10

PATH_SAVE = "/01"

# main ----------
population = generate_population(pop_size, bits)

save(PATH_SAVE + '/population_inicial', 'p', bits, population)

pop_fitness = estimate_fitness(population, pop_size, bits, split)


print('INÍCIO PROCESSO EVOLUTIVO')

for i in range(generation):
    
    # após o while a população de filhos é equivalente a população anterior
    while(int(len(pop_children)/bits) != pop_size):
        # executa somatório dos valores de fitness da população
        fitness_sum = sum_fitness(pop_fitness)

        # seleciona os índices de indivíduos aptos ao cruzamento
        index_parent_1 = roulette(pop_size, fitness_sum, pop_fitness)
        index_parent_2 = roulette(pop_size, fitness_sum, pop_fitness)

        children = uniform_crossover_binary(population[index_parent_1], population[index_parent_2], bits, tc)
        
        # executa procedimento de mutação
        children = mutation(children, tm)

        # adiciona filhos para nova população (geração)
        pop_children = np.append(pop_children, children)

    population = np.reshape(pop_children, (pop_size, bits))

    save(PATH_SAVE + '/population_{}'.format(i), 'p', bits, population)
    
    # zera população de filhos
    pop_children = []

    # calcula fitness da geração atual => pop_fitness 
    pop_fitness = estimate_fitness(population, pop_size, bits, split)

    # executa preenchimento dos vetores de média, pior e melhor (fitness)
    average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / pop_size))
    bad_fitness = np.append(bad_fitness, select_bad_fitness(pop_fitness))
    best_fitness = np.append(best_fitness, select_best_fitness(pop_fitness))

print('FIM PROCESSO EVOLUTIVO')

# plotagem de gráfico
# cria figura e box
fig, ax = plt.subplots()  
# plota as curvas de desempenho
ax.plot(best_fitness, label='melhor')
ax.plot(average_fitness, label='médio')
ax.plot(bad_fitness, label='pior')
# configuração legenda
ax.set_xlabel('iteração')
ax.set_ylabel('fitness')
ax.set_title("Processo evolutivo - {} iterações".format(generation))
ax.legend() 
# salva gráfico em diretório específico
plt.savefig(os.getcwd() + PATH_SAVE + '/evolution.png')
# exibe imagem
plt.show()