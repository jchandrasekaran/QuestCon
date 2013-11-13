"""
This file contains the implementation of the Dialog manager that maintains the responses provided by the classifier. It is based on the work done by Leuski and Traum in "NPCEditor: A Tool for Building Question-Answering Characters"
"""

from numpy import *
from random import choice
import trainer_main as tr
import utility as ut
import trainer_reader as rdr
from collections import deque


class dialog_manager:
	
	def __init__(self, memory_size = 4):
		self.conversation_count = 0

		# Initialize a queue with the required memory of past conversations
		self.history = deque('', memory_size)

		# Load the training sets for both states
		self.training_set_state1 = rdr.trainer_reader('Data/training_ML.csv').load()
		
		self.training_map_1 = dict({0:1})
		
		self.training_set_state1 = self._map_multiple_response(self.training_set_state1, self.training_map_1)
	

		#lambda_pi_1 = [0.022637184564677337, 0.0006614527936524828, 0.8151936922826796, 0.009768802605860376, 0.9977721108143799, 0.9676503420836119]
		#self.lambda_pi_1 = [0.5089104109775326, 0.5432925251421259, 0.02218191679541291, 0.8877708494421628, 0.8206554620659708, 0.6351178059195106]
		self.lambda_pi_1 = [0.2517703455341973, 0.024354405309287175, 0.3108059389694587, 0.6223442089112752, 0.28378859405365237, 0.41789353929407375]
		self.lambda_pi_1 = map(lambda x: x/sum(self.lambda_pi_1), self.lambda_pi_1 )
#0.5340087189383883 #0.7204150300022019
	
		# Use the training sets to train two classifiers
		#self.trn_1 = tr.trainer(0.007672586062919887, lambda_pi_1 , self.training_set_state1)
		self.trn_1 = tr.trainer(self.lambda_pi_1 , self.training_set_state1)
		self.trn_1.train()

		# Set threshold for dialogue generation
		self.threshold_1 = float('0.06')#-0.0015
		
	def _map_multiple_response(self, training_set, tmap):
		training_map = dict()
		for (ques, ans) in training_set:
			training_map[ques] = training_map.get(ques, []) + [ans]
		mtrs = []
		for ques, answers in training_map.iteritems():
			tmap[answers[0]] = answers
			mtrs.append((ques, answers[0]))
		return mtrs
			
			
	def get_reply (self, string):
		
		self.trn = self.trn_1
		self.threshold = self.threshold_1
		self.tmap = self.training_map_1
		
		#print self.lambda_pi_1
	
		# Get the classification for 'string'
		res = self.trn.get_classification(string)
		
		# Get the value, reply of highest ranked response
		value = res[ut.key_max_val_dict(res)]
		reply = ut.key_max_val_dict(res)
		response_prior = ''

		# Check if it crosses minimum threshold
		if value >= self.threshold:
			# Check if it has been repeated in history
			# and respond appropriately
			"""
			if reply in self.history:
				response_prior = 'I\'ll repeat myself, '
				#reply = 'I\'ll repeat myself, '+ choice(self.tmap.get(reply, [reply]))
				if response_prior + reply in self.history:
					response_prior = 'I am tired of this conversation, but ' + response_prior
					if response_prior + reply in self.history:
						response_prior = ''
			"""
			# Put the string in history
			self.history.append (response_prior + reply)
			reply = response_prior + choice(self.tmap.get(reply, [reply]))
			return reply

		# In case responses are below the threshold
		else:
			reply = ' I don\'t know anything about this'
			if reply in self.history:
				reply = 'Ask me something else'

			self.history.append(reply)
			return reply
