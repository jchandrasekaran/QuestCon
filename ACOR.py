"""
This file contains the implementation of Ant Colony Optimization Class. 
"""

import numpy as np

class ACO:

	def __init__(self, cFunc, nDim, nAnt, m, q, e, limit = None):
		self.cfunc = cFunc
		self.dim = nDim
		self.nAnt = nAnt
		self.m = m
		self.q = q
		self.e = e
		if(limit == None):
			self.limit = [[-float('inf'), float('inf')] for i in range(self.dim)]
		else:
			self.limit = limit

		T = []
		for i in range(nAnt): 
			l = []
			for j in range(self.dim):
				l.append((np.random.random() * (self.limit[j][1] - self.limit[j][0]) + self.limit[j][0]))
			T.append( [self.cfunc(l), l] )

		self.T = np.array ([tuple(i) for i in T],dtype=[('fitness',float),('point',(float, self.dim))])
		self.T.sort()
		

	def optimizeMin(self, iterator = 10000):
		global np
		globalTemp = iterator
		for it in range(iterator):
			T = []
			for nm in range(self.m):
				so = 500 - self.T['fitness']
				s = sum(so)

				s = np.random.random_integers(s)
				j = 0
				while s > 0 and j < (self.nAnt - 1) :
					s -= so[j]
					j += 1
			

				ps = []
				for i in range(self.dim):
					
					res = np.random.normal(self._mean(i,j), self._SD(i,j) * (np.exp(-(iterator - globalTemp)/ iterator))) 
					
					if (res < self.limit[i][0]) :
						res = self.limit[i][1] - ((self.limit[i][0] - res) % (self.limit[i][1] - self.limit[i][0]))
					if res > self.limit[i][1] :
						res = self.limit[i][0] + ((res - self.limit[i][1]) % (self.limit[i][1] - self.limit[i][0]))
					ps.append(res)

				T.append( [self.cfunc(ps), ps] )
			globalTemp -= 1
			T = np.array ([tuple(i) for i in T],dtype=[('fitness',float),('point',(float, self.dim))])


			tempArr = np.concatenate([self.T, T], axis = 0 )
			tempArr.sort()
			#print tempArr
			tempArr.resize(self.nAnt)

			self.T = tempArr
			print self.T[0]

		return self.T

	def _weight(self, j):
		global np
		wj = 1 / (self.q * self.nAnt * np.sqrt(2 * np.pi) )
		tmp = (np.exp( -( (self.T['fitness'][j] - 1)**2 / (2 * self.q**2 * self.nAnt**2) ) ) + 0.000000001)
		wj *= tmp
		return wj
	
	def _SD(self, i, j):
		j = int(j)
		SD = 0.00000001
		for r in range(self.nAnt):
			SD += ( abs(self.T['point'][r][i] - self.T['point'][j][i]) / (self.nAnt - 1) )
		return self.e * SD

	def _mean(self, i, j):
		j = int(j)
		return self.T['point'][j][i]

	def _gaussianKernel(self,i, X):
		global np
		s = 0
		for j in range(self.nAnt):
			tmp = self._weight(j)
			tmp *= ( 1/( self._mean(i,j) * np.sqrt(2 * np.pi) ) )
			tmp *= np.exp( -( np.power(X - self._mean(i,j), 2 ) / ( 2 * np.power(self._SD(i,j), 2) ) ))
			s += tmp
		return s	


