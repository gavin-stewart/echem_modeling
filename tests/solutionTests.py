# Tests for the conversion, solutionTools, and dataFileIO

import unittest
import tools.solutionTools as st
import tools.gridTools as gt
import tools.dataFileIO as dfio
import tools.conversion as conv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as mpl
import os.path 
import os
from scipy.stats.distributions import norm

noExpensive = False #Set False to run expensive tests

class ConversionTests(unittest.TestCase):
	def testTimeConversion(self):
		"""Verify that the time conversion function works"""
		EStart = -1
		ERev = 1
		t = np.linspace(0, 1, 100)
		nu = 2./100.
		temp = 25.+273.15
		tau = conv.timeToNondimVoltage(temp, nu, EStart, ERev, t)
		self.assertEqual(tau[-1], conv.nondimPot(temp, EStart) +\
		 2. * conv.nondimPot(temp, ERev - EStart))

class SolutionToolsTests(unittest.TestCase):
	@unittest.skipIf(noExpensive, "Expensive")
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
	
	@unittest.skipIf(noExpensive, "Expensive")
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

	@unittest.skipIf(noExpensive, "Expensive")
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
			


	@unittest.skipIf(noExpensive, "Expensive")
	def testMatchesMartinsData(self):
		"""The solveIDimensional method matches data given by Martin"""
		fileName = './files/dataMartin'

		jsonFileName = "./files/simulationParameters.json"
		dataName = "Martin's experiment"

		t,IObs =  dfio.readTimeCurrentData(fileName)
		
		params = dfio.readDimensionalParametersFromJSON(jsonFileName, dataName)
	
		I, amt = st.solveIFromJSON(t, params)
		
		n = len(t)
		lowerBdd = int(np.ceil(0.2*n))
		upperBdd = int(np.floor(0.8*n))
		ind = range(lowerBdd, upperBdd)
		
		#Check that model accounts for >= 95% of variation about the mean.
		self.assertTrue(np.sum((IObs[ind] - I[ind])**2)/len(IObs) <=\
		5e-2 * np.sum((IObs[ind] - np.mean(IObs[ind])**2)))

	def testExtractHarmonicCorrectlyIDsHarmonics(self):
		"""Pure sines match their first harmonic"""
		x = np.linspace(0,1,1e4)
		freq = 5
		y = np.sin(2*np.pi*freq*x)
		y2 = st.extractHarmonic(1, freq, y)
		fourier = np.fft.rfft(y)
		fourier[freq] = 0
		# since y is a pure tone, it should equal its first harmonic
		self.assertTrue(np.sum(np.square(y2-y))/len(y) < 1e-5)

	def testExtractHarmonicVanishesForNullHarmonics(self):
		"""Pure functions do not have higher harmonics"""
		x = np.linspace(0,1,1e4)
		freq = 5
		y = np.sin(2*np.pi*freq*x)
		y2 = st.extractHarmonic(2, freq, y)
		fourier = np.fft.rfft(y)
		fourier[freq] = 0
		errorEst = np.amax(abs(fourier)) * 10/np.sqrt(2)
		# since y is a pure tone, its second harmonic should vanish
		self.assertTrue(np.sum(np.square(y2))/len(y) < 1e-5)

	def testExtractHarmonicReturnsRealForRealInput(self):
		"""Real functions have real harmonics"""
		y = np.random.rand(1e4)
		y2 = st.extractHarmonic(3, 5, y)
		# Harmonic should be real for real input data
		self.assertTrue(np.isclose(np.amax(np.imag(y2)), 0))

	def testSolutionHarmonics1(self):
		"""Second harmonic roughly matches the results of Morris et al 2015"""

		n = 7e4
		t=np.linspace(0,7,n)
		E_0 = 0
		dE = 80e-3
		freq = 72
		k_0 = 1000
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
	
		I, amt = st.solveIDimensional(t, E_0, dE, freq, k_0, 
		Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, nu, area, 
		coverage, False)


		secHarm = st.extractHarmonic(2, freq * 7, I)
		secHarmMax = np.amax(secHarm)
		self.assertTrue(secHarmMax > 0.9e-4)
		self.assertTrue(secHarmMax < 1.3e-4)
	
	def testSolutionHarmonics2(self):
		"""Fifth harmonic roughly matches Morris et al 2015"""
	
		n = 7e4
		t=np.linspace(0,7,n)
		E_0 = 0
		dE = 80e-3
		freq = 72
		k_0 = 1000
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

		I, amt = st.solveIDimensional(t, E_0, dE, freq, k_0, 
		Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, nu, area, 
		coverage, False)

		import matplotlib.pyplot as plt

		secHarm = st.extractHarmonic(5, freq * 7, I)
		import matplotlib.pyplot as plt
		plt.plot(I)
		plt.show()
		secHarmMax = np.amax(secHarm)
		self.assertTrue(secHarmMax > 1e-5)
		self.assertTrue(secHarmMax < 2.e-5)

	def testShortCenteredKaiserWindowFailsForCentersBelowZero(self):
		hl = 3
		center = -5
		N = 10
		with self.assertRaises(ValueError):
			st.shortCenteredKaiserWindow(hl, center, N)

	def testShortCenteredKaiserWindowFailsForCentersAboveN(self):
		hl = 3
		center = 11
		N = 10
		with self.assertRaises(ValueError):
			st.shortCenteredKaiserWindow(hl, center, N)

	def testShortCenteredKaiserWindowRunForLeftBdyBelow0(self):
		hl = 5
		center = 4
		N = 10
		st.shortCenteredKaiserWindow(hl,center,N)

	def testShortCenteredKaiserWindowRunsForRightEndpointAboveN(self):
		hl = 5
		center = 6
		N = 10
		st.shortCenteredKaiserWindow(hl, center, N)

	def testShortCenteredKaiserWindowVanishesOutisdeSupport(self):
		self.assertEqual(st.shortCenteredKaiserWindow(5, 1, 10)[9], 0)
		self.assertEqual(st.shortCenteredKaiserWindow(5, 9, 10)[0], 0)

	def testUnifSpaceParamBasic(self):
		pts, weights = gt.unifSpacedParam(15, -7.0, 7.0, 0, 1, False)
		self.assertEqual(list(pts), list(np.linspace(-7,7,15)))
		for pt, weight in zip(pts, weights):
			if np.isclose(pt, -7):
				self.assertTrue(np.isclose(weight, norm.cdf(-6.5)))
			elif np.isclose(pt, 7):
				self.assertTrue(np.isclose(weight, norm.sf(6.5)))
			else:
				self.assertTrue(np.isclose(weight, norm.cdf(pt+0.5) -norm.cdf(pt-0.5)))

	def testUnifSpaceParamsWithShiftAndStretch(self):
		pts, weights = gt.unifSpacedParam(15, -7.0, 7.0, 1, 2, False)
		for pt, weight in zip(pts, weights):
			if np.isclose(pt,-7.0):
				self.assertTrue(np.isclose(weight, norm.cdf(-6.5, loc=1, scale=2)))
			elif np.isclose(pt,7.0):
				self.assertTrue(np.isclose(weight, norm.sf(6.5, loc=1, scale=2)))
			else:
				self.assertTrue(np.isclose(weight, norm.cdf(pt+0.5, loc=1, scale=2) -norm.cdf(pt-0.5, loc=1, scale=2)))


	def testUnifSpaceProbNorm(self):
		vals = gt.unifSpacedProb(15, 0, 1, False)
		self.assertEqual(len(vals[0]), 15)
		self.assertEqual(len(list(vals[1])), 15)
		for k in vals[1]:
			self.assertTrue(np.isclose(1/15.0, vals[k]))	

	@unittest.skip("Expensive")
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
