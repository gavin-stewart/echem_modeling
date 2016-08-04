
import tools.solutionTools as st
import tools.gridTools as gt
import tools.io as io
import numpy as np
from scipy.stats.distributions import norm
import unittest

def k0QuadFunFactory(kSD):
	return lambda n: gt.hermgaussParam(n, 4e3, kSD, True)

def E0QuadFunFactory(ESD):
	return lambda n: gt.hermgaussParam(n, -0.41, ESD, False)

class DispersionCoulombTests(unittest.TestCase):
	""" 
	Tests that functions relating to dispersion perform as expected.	
	"""

	ESDVals = [1e-3, 1e-2, 1e-1]
	kSDVals = [1, 2, 3]
	numSampPts = 15
	
	def setUp(self):
		self.baseData = io.readParametersFromJSON('./files/simulationParameters.json', "disp Coulomb")
		tEnd = (self.baseData["pot_rev"] - self.baseData["pot_start"]) / self.baseData["nu"]
		self.num_time_pts = np.ceil(tEnd * 8.959 * 200)
	
		self.time_step = tEnd / (self.num_time_pts - 1)

	def addResistance(self):
		self.baseData["Ru"] = 100.0

	def addConstCapacitance(self):
		self.baseData["Cdl"] = 1e-4

	def addVarCapacitance(self):
		self.addConstCapacitance()
		self.baseData["Cdl1"] = 6e-4
		self.baseData["Cdl2"] = 2.5e-4
		self.baseData["Cdl3"] = 1.1e-6

	def addAC(self):
		self.baseData["freq"] = 8.959
		self.baseData["ac_amplitude"] = 150e-3

	def checkSatisfiesCoulomb(self):
		I, amtNoDisp = st.solve_reaction_from_json(self.time_step, self.num_time_pts, self.baseData)
		endAmt = amtNoDisp[-1]
		INoDispInt = np.sum(I) / self.num_time_pts
		
		for ESD in self.ESDVals:
			for kSD in self.kSDVals:
				self.baseData["bins"] = gt.productGrid(E0QuadFunFactory(ESD), self.numSampPts, k0QuadFunFactory(kSD), self.numSampPts)
				I, amt = st.solve_reaction_from_json(self.time_step, self.num_time_pts, self.baseData)
				self.assertAlmostEqual(endAmt, amt[-1])
				IInt = np.sum(I) / self.num_time_pts
				self.assertAlmostEqual(IInt, INoDispInt)

	def testDispersionObeysCoulombsLawDCWithNoResistanceNoCapacitance(self):
		"""Ensure that the integral of Faradaic current over time is constant."""
		self.checkSatisfiesCoulomb()

	
	def testDispersionObeysCoulombsLawDCWithNoCapacitance(self):
		"""Ensure that the integral of Faradaic current over time is constant."""
		self.addResistance()
		self.checkSatisfiesCoulomb()

	def testDispersionObeysCoulombsLawDCWithNonVariableCapacitance(self):
		"""Ensure that the integral of Faradaic current over time is constant."""
		self.addResistance()
		self.addConstCapacitance()
		self.checkSatisfiesCoulomb()

	def testDispersionObeysCoulombsLawDCWithVariableCapacitance(self):
		"""Ensure that the integral of Faradaic current over time is constant."""
		self.addResistance()
		self.addVarCapacitance()
		self.checkSatisfiesCoulomb()

	def testDispersionObeysCoulombsLawACWithNoResistanceNoCapacitance(self):
		self.addAC()
		self.checkSatisfiesCoulomb()

	def testDispersionObeysCoulombsLawACWithNoCapacitance(self):
		self.addAC()
		self.addResistance()
		self.checkSatisfiesCoulomb()

	def testDispersionObeysCoulombsLawACWithConstCapacitance(self):
		self.addAC()
		self.addConstCapacitance()
		self.addResistance()
		self.checkSatisfiesCoulomb()

	def testDispersionObeysCoulombsLawACWithVariableCapacitance(self):
		self.addAC()
		self.addVarCapacitance()
		self.addResistance()
		self.checkSatisfiesCoulomb()
