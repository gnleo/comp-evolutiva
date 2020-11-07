#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:39:27 2020

@author: gnleo
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt

def fitness(subject, distances):
	distances_sum = 0
	for i in range (len(subject)-1):
#		print("de = {}, para = {} | distance = {}".format(subject[i]-1,subject[i+1]-1, distances[subject[i]-1,subject[i+1]-1]))
#	print("de = {}, para = {} | distance = {}".format(subject[len(subject)-1]-1,subject[0]-1, distances[subject[len(subject)-1], subject[0]-1]))
		distances_sum += distances[subject[i]-1, subject[i+1]-1]
	distances_sum += distances[subject[len(subject)-1]-1, subject[0]-1]
	return 1/distances_sum



#forma aleatoria
#distances = np.random.random((3, 3))
#distribuicao uniforme
distances = np.random.uniform(0.1,10,(5,5))
for i in range(len(distances)):
	distances[i,i] = 0
	


popsize=100
ngeneration=1000
k_children = 5
ncities = len(distances)

population = []
r_fitness = np.zeros(popsize)

for i in range(popsize):
	iterable = np.arange(1, ncities+1, 1)
	subject = rd.sample(list(iterable), ncities)
	# cria populacao
	population = np.append(population, subject)
	# preenche vetor fitness
	r_fitness[i] = fitness(subject, distances)
population = np.reshape(population, (popsize, ncities))


output = np.empty([ngeneration, 3])
for i in range(ngeneration):
	children = []
	children_fitness = np.zeros(k_children)
	
	# gera os filhos
	for j in range(k_children):
		iterable_population = np.arange(1, popsize+1, 1)
		parentId = rd.sample(list(iterable_population), 1)
		
		new = population[parentId[0]-1].astype('int')
		new = np.reshape(new, (1, k_children))
		
		selected = rd.sample(list(iterable), 2)
		aux = int(new[0][selected[0]-1])
		new[0][selected[0]-1] = int(new[0][selected[1]-1])
		new[0][selected[1]-1] = aux
		
		children_fitness[j] = fitness(new[0], distances)
		
		children = np.append(children, new)
		
	children = np.reshape(children, (k_children, k_children))

	# competicao
	for j in range(k_children):
		id_P = rd.sample(list(iterable), 1)
		if(r_fitness[id_P] < children_fitness[j]):
			population[id_P] = children[j]
			r_fitness[id_P] = children_fitness[j]
			
	output[i][0] = i
	output[i][1] = np.mean(r_fitness)
	output[i][2] = np.std(r_fitness)
			
#fitness(subject, distances)
fig, a = plt.subplots()
a.plot(output[0:, 0], output[0:, 1], label='media')
a.legend()
fig, b = plt.subplots()
b.plot(output[0:, 0], output[0:, 2], label='dp')
b.legend()
