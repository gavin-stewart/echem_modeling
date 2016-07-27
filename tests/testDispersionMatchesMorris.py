import tools.solutionTools as st
import tools.gridTools as gt
import tools.dataFileIO as dfio
import numpy as np
from scipy.stats.distributions import norm
import unittest

class MatchesMorrisTestSet(unittest.TestCase):
	
	#TODO: Clean up.

	def testsolveIWithDispersionMatchesMorrisNoDisp(self):
		"""Verify the solveIWithDispersionDimensional method reproduces the results of Morris et al 2015"""
		t = np.linspace(0,7, 7e3)

		fileName = "./files/simulationParameters.json"
		dataName = "Morris no disp"
		data = dfio.readDimensionalParametersFromJSON(fileName, dataName)
		I, amt = st.solveIFromJSON(t, data)

		self.assertTrue(np.amax(I) < 8e-7)
		self.assertTrue(np.amax(I) > 7e-7)

		width = st.widthAtHalfMaximum(I, t,data["nu"])

		self.assertTrue(abs(width - 0.123) < 7e-4)
	
	def testsolveIWithDispersionMatchesMorrisE0Disp(self):
		"""The solveIWithDispersionDimensional method reproduces the results fo Morris et al 2015"""
		time = np.linspace(0, 7, 7e4) #1000 pts per second
		dE = 0
		freq = 0
		Ru = 0
		Cdl = 0.
		Cdl1 = 0.
		Cdl2 = 0.
		Cdl3 = 0.
		EStart = -0.2
		ERev = 0.5
		temp = 293
		nu = 0.1
		area = 1
		coverage = 1e-11
		E_0BinsUnscaled = np.linspace(-17.5e-3, 17.5e-3, 15)
		#Define wE in terms of bin widths
		leftBinEnds = np.empty(15)
		leftBinEnds[1:] = np.linspace(-16.25e-3, 16.25e-3, 14)
		leftBinEnds[0] = -np.inf
		rightBinEnds = np.empty(15)
		rightBinEnds[:-1] = np.linspace(-16.25e-3, 16.25e-3, 14)
		rightBinEnds[-1] = np.inf
		wE = norm.cdf(rightBinEnds, loc=0, scale=5e-3) -\
		norm.cdf(leftBinEnds, loc=0, scale=5e-3)
		
		k_0Bins = {0.1 : 1.0}
		expWidth = np.array([0.124, 0.126, 0.129, 0.133, 0.138, 0.144,
		0.151, 0.159, 0.167, 0.176])
		
		self.assertEqual(np.sum(wE), 1)

		for l,ew in zip(range(1, 11), expWidth):
			E_0Vals = l * E_0BinsUnscaled
			self.assertTrue(np.isclose(E_0Vals[-1]-E_0Vals[0], l*35.e-3))
			bins = [(E_0, 0.1, we) for E_0, we in zip(E_0Vals, wE)]
			
			I, amt = st.solveIWithDispersionDimensionalBins(time, dE, 
			freq, Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, 
			temp, nu, area, coverage, bins, False)
		
			width = st.widthAtHalfMaximum(I, time, nu)
			self.assertTrue(abs(width - ew) < 7e-4) #Rounding error + 2*step size + solution error (estimated at 1*step size)

	def testSolveIWithDispersionMatchesMorrisk0Disp(self):
		"""The solveIWithDispersionDimensional method matches results from Morris et al 2015"""
		time = np.linspace(0, 7, 7e4) #1000 pts per second
		dE = 0
		freq = 0
		Ru = 0
		Cdl = 0
		Cdl1 = 0
		Cdl2 = 0
		Cdl3 = 0
		EStart = -0.2
		ERev = 0.5
		temp = 293
		nu = 0.1

		area = 1
		coverage = 1e-11
		k_0BinsUnscaled = np.linspace(-7, 7, 15)
		#Define wE in terms of bin widths
		leftBinEnds = np.empty(15)
		leftBinEnds[1:] = np.linspace(-6.5, 6.5, 14)
		leftBinEnds[0] = -np.inf
		rightBinEnds = np.empty(15)
		rightBinEnds[:-1] = np.linspace(-6.5, 6.5, 14)
		rightBinEnds[-1] = np.inf
		wK = norm.cdf(rightBinEnds, loc=0, scale=2) -\
		norm.cdf(leftBinEnds, loc=0, scale=2)
		
		expWidth = np.array([0.124, 0.128, 0.134, 0.141, 0.150, 0.161,
		0.172, 0.185, 0.198, 0.211])

		self.assertEqual(np.sum(wK), 1)
		
		for m,ew in zip(range(1, 11), expWidth):
			k_0Vals = 0.1 * 2**(0.1*m*k_0BinsUnscaled)
			bins = [(0, k_0, w) for k_0, w in zip(k_0Vals, wK)]
			
			I, amt = st.solveIWithDispersionDimensionalBins(time, dE, 
			freq, Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, 
			temp, nu, area, coverage, bins, False)
		
			width = st.widthAtHalfMaximum(I, time, nu)
			err = abs(width - ew)
			#Rounding error + 2*step size + error in I 
			#(estimated at 1*stepsize)
			self.assertTrue(err < 7e-4) 
 	
	@unittest.expectedFailure
	def testMorrisE0DispWithMC(self):
		fileName = './files/simulationParameters.json'
		dataName = 'morris-E0-disp-MC'
		data = dfio.readParametersFromJSON(fileName, dataName)
		t = np.linspace(0,7,7e3) #1000 pts per second.
		I, amt = st.solveIFromJSON(t, data)
		expWidth = 0.176
		width = st.widthAtHalfMaximum(I, t, data["nu"])
		self.assertTrue(abs(expWidth - width) < 5e-3)


