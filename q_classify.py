"""
This file contains the implementation of Question Classification as described in the paper by Gaurav Aggarwal, Richa Bhayani and Tejaswi Tenneti.
It implement NaiveBayes Classifier and returns noth the fine and coarse classes for the question. 
"""

import nltk
import random
from nltk.classify import NaiveBayesClassifier
import csv
import numpy as np
import utility as ut
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


class q_classification:

	def __init__ (self, path = "Data/Dataset1.csv"):
		cr = csv.reader(open(path,"rb"))
		temp = [(row[1], row[0]) for row in cr]
		
		self.tag_list = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
		self.data_set_total = []
		
		self.data_set_total = [(self.feature_extractor(ut.clean(n)), g) for (n,g) in temp]
		self.train_set = self.data_set_total
		self.train()
		
	def train(self):
		self.classifier = NaiveBayesClassifier.train(self.train_set)
		
	def classify(self,Xtest):
		
		Prediction = list()
		
		for x in Xtest:
			Prediction.append(self.classifier.classify(self.feature_extractor(ut.clean(x))))
		
		return Prediction
	
	def feature_extractor(self,sentence):
		features = {}
		
		words = word_tokenize(ut.clean(sentence))
		features["length"] = len(words)
		
		for q in ["how","where","when","what","why"]:
			features["has(%s)" % q] = (q in words)
		
		temp = [ (b) for (a,b) in pos_tag(words)]
		
		tag_dict = dict ()
		
		for tag in temp:
			tag_dict [tag] = tag_dict.get(tag,0) + 1
		
		for tags in self.tag_list.classifier().labels():
			features["count(%s)" % tags] = tag_dict.get(tags,0)
        	#features["has(%s)" % tags] = (tag_dict.get(tags,0) != 0)
		
		return features
		
"""
path = "Data/Dataset1.csv"

obj = q_classification(path)

obj.train()
 

print ''
s ='are you coming'
print s
print obj.classify([s])
"""

