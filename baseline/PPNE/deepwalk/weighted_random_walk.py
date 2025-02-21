import networkx as nx
import numpy as np
import random
import time
from pdb import set_trace as st

def getTransitionMatrix(network, nodes):
	nodes = list(nodes)
	# adj_mat = nx.adjacency_matrix(network)
	matrix = np.empty([len(nodes), len(nodes)])

	for i in range(0, len(nodes)):
		neighs = list(network.neighbors(nodes[i]))
		sums = 0
		for neigh in neighs:
			sums += network[nodes[i]][neigh]['weight']

		for j in range(0, len(nodes)):
			# if i == j :
			# 	matrix[i,j] = 0
			# else:
			if nodes[j] not in neighs:
				matrix[i, j] = 0
			else:
				matrix[i, j] = network[nodes[i]][nodes[j]]['weight'] / float(sums)
	# adj_mat = adj_mat / np.linalg.norm(adj_mat, axis=1, keepdims=True)

	return matrix

def getTransitionMatrix_fast(network):
	matrix = nx.adjacency_matrix(network).toarray()
	row_sum = np.sum(matrix, axis=1, keepdims=True)
	return matrix / row_sum

def generateSequence(startIndex, transitionMatrix, path_length, alpha):
	result = [startIndex]
	current = startIndex

	for i in range(0, path_length):
		if random.random() < alpha:
			nextIndex = startIndex
		else:
			probs = transitionMatrix[current]
			nextIndex = np.random.choice(len(probs), 1, p=probs)[0]

		result.append(nextIndex)
		current = nextIndex

	return result

def random_walk(G, num_paths, path_length, alpha):
	nodes = list(G.nodes())
	transitionMatrix = getTransitionMatrix_fast(G)

	sentenceList = []

	for i in range(0, len(nodes)):
		for j in range(0, num_paths):
			indexList = generateSequence(i, transitionMatrix, path_length, alpha)
			sentence = [str(nodes[tmp]) for tmp in indexList]
			sentenceList.append(sentence)

	return sentenceList