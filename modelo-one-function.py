# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:47:51 2020

@author: gnleo
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt

def fitness(nprocess, subject, capacity, cost=1000):
#	MIPS
#	1000 800 700
#	25 processos numero
#	10 8 7 alocacoes do processo
	v = np.zeros(len(capacity))
	for i in range(nprocess):
		coreId = subject[0][i]
		coreId = int(coreId)
		v[coreId -1] = v[coreId -1] + cost / capacity[coreId -1]
	return (1/max(v))
	

def model_one(capacity, nprocess=5, popsize=10, ngenerations= 15):
	# gera M individuos
	ncores = len(capacity)
	population = []
	output = np.empty([ngenerations, 3])
	
	for i in range(popsize):
		element = []
		for j in range(nprocess):
			iterable = np.arange(1, ncores +1 , 1)
			core = rd.sample(list(iterable), 1)
			element = np.append(element, core)
		population = np.append(population, element)
	population = np.reshape(population, (popsize, nprocess))
	
	for i in range(ngenerations):
		# clone de pai
		parentId = rd.sample(list(np.arange(1, popsize, 1)), 1)
		new = population[parentId]
		# realizando mutacao no filho
		mutation_geneId = rd.sample(list(np.arange(1, nprocess, 1)), 1)
		new[0][mutation_geneId] = rd.sample(list(iterable), 1)
		# selecionar um elemento aleatorio para competicao
		selectedId = rd.sample(list(np.arange(1, popsize, 1)), 1)
		
		otimization = fitness(nprocess, new, capacity)
		otimization_selected = fitness(nprocess, population[selectedId], capacity)
		if(otimization > otimization_selected):
			population[selectedId] = new
			
		# media fitness
		fit = np.zeros(len(population))
		for j in range(len(population)):
			fit[j] = fitness(nprocess, population[[j]], capacity)
		
#		print("{} {} {}".format(i, np.mean(fit), np.std(fit)))
		output[i][0] = i
		output[i][1] = np.mean(fit)
		output[i][2] = np.std(fit)
		
	return output
		
t = model_one(capacity=[1000,800,700], ngenerations=1000, nprocess=25, popsize=100)

fig, a = plt.subplots()
a.plot(t[0:, 0], t[0:, 1], label='media')
a.legend()
fig, b = plt.subplots()
b.plot(t[0:, 0], t[0:, 2], label='dp')
b.legend()
