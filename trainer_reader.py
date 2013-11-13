"""
This file contains the implementation of the reader reading the question answering data and preprocessing it.
"""

import numpy as np
import utility as ut
import csv

class trainer_reader:

	# Constructor to initialize the reader	
	def __init__ (self, filename, format='csv'):
		self.format = format
		self.filename = filename
		self.data = None

	def load(self, useCache = True):
		# Implement Caching for multiple calls to load
		if(useCache and self.data != None):
			return self.data
	
		# Check Format [Only CSV is accepted as of now]
		if(self.format == 'csv'):

			# IF first call, then load data and format it
			self.data = []			
			with open(self.filename, 'rb') as csvfile:
				# Read the file, escaped commas with \
				data_reader = csv.reader(csvfile, delimiter = ',', escapechar = '\\')
				for row in data_reader:	
					# Strip Spaces from the data	
					row = np.char.strip(row)

					# Format as required by learner
					self.data.append( (row[0], row[1]) )
	
			return self.data
		else:
			raise Exception('Unsupported Format give to trainer_reader')



