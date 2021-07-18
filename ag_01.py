""" @author: gnleo """

# library's --------
from ag_functions import *
import matplotlib.pyplot as plt

# variables ----------
population = []
children = []
pop_children = []
average_fitness = []
bad_fitness = []
best_fitness = []
pop_fitness = np.zeros(POP_SIZE)
children_fitness = np.zeros(POP_SIZE)

PATH_SAVE = "/01/cruzamento_uniforme"

# main ----------
population = generate_population()

save(PATH_SAVE + '/population_inicial', 'p', population)

pop_fitness = estimate_fitness(population)


print('INÍCIO PROCESSO EVOLUTIVO')

for i in range(GENERATION):
    
    # após o while a população de filhos é equivalente a população anterior
    while(int(len(pop_children)/BITS) != POP_SIZE):
        # executa somatório dos valores de fitness da população
        fitness_sum = sum_fitness(pop_fitness)

        # seleciona os índices de indivíduos aptos ao cruzamento
        index_parent_1 = roulette(fitness_sum, pop_fitness)
        index_parent_2 = roulette(fitness_sum, pop_fitness)

        children = uniform_crossover_binary(population[index_parent_1], population[index_parent_2])
        
        # executa procedimento de mutação
        children = mutation(children)

        # adiciona filhos para nova população (geração)
        pop_children = np.append(pop_children, children)

    population = np.reshape(pop_children, (POP_SIZE, BITS))

    save(PATH_SAVE + '/population_{}'.format(i), 'p', population)
    
    # zera população de filhos
    pop_children = []

    # calcula fitness da geração atual => pop_fitness 
    pop_fitness = estimate_fitness(population)

    # executa preenchimento dos vetores de média, pior e melhor (fitness)
    average_fitness = np.append(average_fitness, ((sum_fitness(pop_fitness)) / POP_SIZE))
    
    # MAXIMIZAÇÃO
    # bad_fitness = np.append(bad_fitness, select_bad_fitness(pop_fitness))
    # best_fitness = np.append(best_fitness, select_best_fitness(pop_fitness))
    
    # MINIMIZAÇÃO
    bad_fitness = np.append(bad_fitness, select_best_fitness(pop_fitness))
    best_fitness = np.append(best_fitness, select_bad_fitness(pop_fitness))

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
ax.set_title("Processo evolutivo - {} iterações".format(GENERATION))
ax.legend() 
# salva gráfico em diretório específico
plt.savefig(os.getcwd() + PATH_SAVE + '/evolution.png')
# exibe imagem
plt.show()