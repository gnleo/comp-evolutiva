#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:47:15 2020

@author: gnleo
"""

import numpy as np
import random as rd

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

capacity = [1000,800,700]
nprocess=5
ncores= len(capacity)
popsize=10
ngenerations= 15
population = []

for i in range(popsize):
	element = []
	for j in range(nprocess):
		iterable = np.arange(1, ncores +1 , 1)
		core = rd.sample(list(iterable), 1)
		element = np.append(element, core)
	population = np.append(population, element)
population = np.reshape(population, (popsize, nprocess))


for i in range(ngenerations):
	print("GERACAO {}".format(i))
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
	
	print("FITNESS")
	print("Média = {}".format(np.mean(fit)))
	print("Desvio padrão = {}".format(np.std(fit)))
