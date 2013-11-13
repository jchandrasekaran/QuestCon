"""
This file contains the implementation of Optimization of parameters for the Question-Answering system  
"""

import trainer_main as tr
import utility as ut
import numpy as np
import ACOR
import trainer_reader as rdr

def map_multiple_response(training_set, tmap):
	training_map = dict()
	for (ques, ans) in training_set:
		training_map[ques] = training_map.get(ques, []) + [ans]
	mtrs = []
	for ques, answers in training_map.iteritems():
		tmap[answers[0]] = answers
		mtrs.append((ques, answers[0]))
	return mtrs

tmap = dict()
reader = rdr.trainer_reader('Data/training_ML.csv')
training_set = reader.load()

training_set = map_multiple_response(training_set, tmap)

print np.sort(training_set),'\n'

CM = tr.OptimizeLambda(training_set)
Optimization = ACOR.ACO(cFunc = CM.costFunction, nDim = 6, nAnt = 50, m = 20, q = 2.14, e = 3.5, limit = CM.getLimits())


Optimization.optimizeMin(500)

