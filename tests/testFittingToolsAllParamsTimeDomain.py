import tools.fittingTools as ft
import tools.solutionTools as st
import tools.io as io
import numpy as np
import unittest
import matplotlib.pyplot as plt
import copy

class FittingToolsAllParamsTimeDomainTestCase(unittest.TestCase):
    """"
    Test if the capacitance and frequency parameters can be recovered when E_0
    and k_0 are also fit in the time domain.
    """

    @classmethod
    def setUpClass(cls):
        # Read in data
	fileName = "./files/simulationParameters.json"
	dataName = "Martin's experiment"
	cls.params = io.readParametersFromJSON(fileName, dataName)
	cls.tEnd = 2 * (cls.params["ERev"] - cls.params["EStart"]) / cls.params["nu"]
	cls.t = np.linspace(0, cls.tEnd, int(np.ceil(cls.tEnd * cls.params["freq"] * 200)))
	cls.IObs, _ = st.solveIFromJSON(cls.t, cls.params)


    def setUp(self):
	params = self.params
        self.exp = ft.ACExperiment(self.IObs, params["Ru"], params["temp"], 
                      params["area"], params["coverage"], params["nu"], 
                      9, params["dE"], params["EStart"], 
                      params["ERev"], self.tEnd, params["reverse"])
	self.exp.addDebugParam("loud-cma")
        print '' #Newline

    def tearDown(self):
        pass

    def setE0k0ToCorrect(self):
	self.exp.E_0 = self.params["E_0"]
        self.exp.k_0 = self.params["k_0"]

    def setE0ToCorrect(self):
        self.exp.E_0 = self.params["E_0"]

    def setk0ToCorrect(self):
        self.exp.k_0 = self.params["k_0"]

    def setSliceToWholeTime(self):
        self.exp.noFaradaicSlices = [np.s_[:]]
        

    def checkMatchesParams(self, *args, **kwargs):
        exp = self.exp
        exp.fitAllParamsTimeDomain(*args, **kwargs)
	self.assertAlmostEqual(exp.Cdl, self.params["Cdl"], 6)
	self.assertAlmostEqual(exp.Cdl1, self.params["Cdl1"], 6)
	self.assertAlmostEqual(exp.Cdl2, self.params["Cdl2"], 6)
	self.assertAlmostEqual(exp.Cdl3, self.params["Cdl3"], 7)
	self.assertAlmostEqual(exp.freq, self.params["freq"], 3)

    def testFitMatchesActualOverNormalSliceNothingGiven(self):
	exp = self.exp
        self.setE0k0ToCorrect()
        self.checkMatchesParams()

    def testFitFunctionMatchesActualNormalSliceOnlyCapacitanceGivenNoE0(self):
        params = self.params
        self.setk0ToCorrect()
        self.checkMatchesParams(self.params["Cdl"], self.params["Cdl1"], self.params["Cdl2"], self.params["Cdl3"])

    def testFitMatchesActualOverNormalSliceNothingGivenNoE0(self):
	exp = self.exp
        self.setk0ToCorrect()
        self.checkMatchesParams()

    def testFitFunctionMatchesActualNormalSliceOnlyCapacitanceGivenNoE0k0(self):
        params = self.params
        self.checkMatchesParams(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

    def testFitMatchesActualOverNormalSliceNothingGivenNoE0k0(self):
        self.checkMatchesParams()
