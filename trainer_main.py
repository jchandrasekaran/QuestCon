#Notes
"""
This file contains the implementation of the classification algorithm that ranks the responses based on KL-Divergence Values. It is based on the work done by Leuski and Traum in "A STATISTICAL APPROACH FOR TEXT PROCESSING IN VIRTUAL HUMANS"
"""
import numpy as np
import utility as ut
import nltk

class trainer:

	def __init__ (self,lambda_pi, training_set):
		
		self.lambda_pi = lambda_pi
		
		# Clean (Lower Case; Remove Punctuation) Training Set
		# Keep backup of formatted data as well
		self.training_orig = dict()
		self.training_set = []
		for (ques, ans) in training_set:
			self.training_set.append((ut.clean(ques),ut.clean(ans)))
			self.training_orig[ut.clean(ans)] = ans

		
		self.unigram_dict = dict()
		self.bigram_dict = dict()
		self.trigram_dict = dict()
		self.unigram_tot_dict = dict()
		self.bigram_tot_dict = dict()
		self.trigram_tot_dict = dict()
		self.len = 0
		self.train()
		

	def train(self):
		
		for (ques, ans) in self.training_set:
			uni = nltk.tokenize.word_tokenize(ques)
			self.len += len(uni)
			for t in uni:
				self.unigram_tot_dict[t] = self.unigram_tot_dict.get(t,0) + 1
				self.unigram_dict[(ques,t)] = self.unigram_dict.get((ques,t),0) + 1
			bi = nltk.bigrams (uni)
			for t in bi:
				self.bigram_tot_dict[t] = self.bigram_tot_dict.get(t,0) + 1
				self.bigram_dict[(ques,t)] = self.bigram_dict.get((ques,t),0) + 1
			tri = nltk.trigrams (uni)
			for t in tri:
				self.trigram_tot_dict[t] = self.trigram_tot_dict.get(t,0) + 1
				self.trigram_dict[(ques,t)] = self.trigram_dict.get((ques,t),0) + 1
			

	def get_classification(self, text):
		text = ut.clean(text)
	
		uni = nltk.tokenize.word_tokenize(text)
		
		bi = nltk.bigrams (uni)
		tri = nltk.trigrams (uni)
		
		temp_lambda = self.lambda_pi
		
		# Map to store answer to its divergence pairs
		list_of_ans = dict()
		
		for (ques, ans) in self.training_set:
			
			fin_val = 0.0
		
			for t in uni:
				fin_val += temp_lambda[5] * (float(self.unigram_tot_dict.get(t,0))/self.len)
				fin_val += temp_lambda[4] * (float(self.unigram_dict.get((ques,t),0))/len(ques))
			
			for t in bi:
				fin_val += temp_lambda[3] * (float(self.bigram_tot_dict.get(t,0))/self.unigram_tot_dict.get(t[:1],1))
				fin_val += temp_lambda[2] * (float(self.bigram_dict.get((ques,t),0))/self.unigram_dict.get((ques,t[:1]),1)) 
			
			for t in tri:
				fin_val += temp_lambda[1] * (float(self.trigram_tot_dict.get(t,0))/self.bigram_tot_dict.get(t[:2],1))
				fin_val += temp_lambda[0] * (float(self.trigram_dict.get((ques,t),0))/self.bigram_dict.get((ques,t[:2]),1))		
			
			list_of_ans[self.training_orig.get(ans, ans)] = fin_val
		
		# Return Weighted list of responses
		return list_of_ans

# OptimizeLambda Class
# Inherits from trainer class, provides an interface to optimize
# lambda values present in trainer. 
# 
# Implements a cost function, that takes in these parameters and returns a cost 
# that is equal to the number of misclassified examples in the training set
# 
# Requires a seperate optimization algorithm implementation
class OptimizeLambda(trainer):

	# Construct class from training set, and random initial lambda
	def __init__(self, training_set):
		trainer.__init__(self, 0.6, training_set)
		self.limits = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
		self.dim = 6

	# Limits of the parameters being considered
	def getLimits (self):
		return self.limits

	# Returns Number of dimensions for optimization
	def getDimensions(self):
		return self.dim
	
	# Defines a cost function that returns the absolute error
	# in classification of training data for given values for
	# lambda_phi and lambda_pi
	def costFunction (self, params):
	
		self.lambda_pi = map(lambda x: x/sum(params), params )
		#print self.lambda_pi

		# Train the classifier
		self.train()

		# Calculate Error
		error = 0.0
		for (ques,ans) in self.training_set:
			res = self.get_classification(ques)
			pred_ans = ut.clean(ut.key_max_val_dict (res))
			# Add one for every misclassified result
			if(pred_ans != ans):
#				print pred_ans, ans
#				raw_input()
				error += 1

		return error
