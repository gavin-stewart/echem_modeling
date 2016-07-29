import tools.fittingTools as ft
import tools.solutionTools as st
import tools.dataFileIO as dfio
import numpy as np
import unittest
import matplotlib.pyplot as plt
import copy

class FittingToolsMaxInformationTestCase(unittest.TestCase):
    """"
    Give the fitting procedure the true parameters as a starting point.
    """

    @classmethod
    def setUpClass(cls):
        # Read in data
	fileName = "./files/simulationParameters.json"
	dataName = "Martin's experiment"
	cls.params = dfio.readParametersFromJSON(fileName, dataName)
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

    def tearDown(self):
        pass

    def setE0k0ToCorrect(self):
	self.exp.E_0 = self.params["E_0"]
        self.exp.k_0 = self.params["k_0"]

    def setSliceToWholeTime(self):
        self.exp.noFaradaicSlices = [np.s_[:]]
        

    def checkMatchesParams(self):
        exp = self.exp 
	self.assertAlmostEqual(exp.Cdl, self.params["Cdl"], 7)
	self.assertAlmostEqual(exp.Cdl1, self.params["Cdl1"], 7)
	self.assertAlmostEqual(exp.Cdl2, self.params["Cdl2"], 7)
	self.assertAlmostEqual(exp.Cdl3, self.params["Cdl3"], 8)
	self.assertAlmostEqual(exp.freq, self.params["freq"], 4)

    def testFitFunctionMatchesActualObjectiveOverWholeTimeWithRealParameters(self):
	exp = self.exp
	params = self.params
        exp.freq = params["freq"]
        self.setSliceToWholeTime()
        self.setE0k0ToCorrect()
        exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

        self.checkMatchesParams()

    def testFitFunctionMatchesActualObjectiveOverWholeTimeOnlyCapacitanceGiven(self):
	exp = self.exp
	params = self.params
        self.setSliceToWholeTime()
        self.setE0k0ToCorrect()
        exp.addDebugParam("No-freq-preprocessing")
        exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

        self.checkMatchesParams()

    def testFitMatchesActualOverWholeTimeNothingGiven(self):
	exp = self.exp
	params = self.params
        self.setSliceToWholeTime()
        self.setE0k0ToCorrect()
        exp.addDebugParam("No-freq-preprocessing")
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()

    def testFitMatchesActualNormalSliceParametersGiven(self):
	exp = self.exp
	params = self.params
        exp.freq = params["freq"]
        self.setE0k0ToCorrect()
        exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

        self.checkMatchesParams()

    def testFitFunctionMatchesActualNormalSliceOnlyCapacitanceGiven(self):
	exp = self.exp
	params = self.params
        self.setE0k0ToCorrect()
        exp.addDebugParam("No-freq-preprocessing")
        exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

        self.checkMatchesParams()

    def testFitMatchesActualOverNormalSliceNothingGiven(self):
	exp = self.exp
	params = self.params
        self.setE0k0ToCorrect()
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()

    def testFitMatchesActualNormalSliceParametersGivenNoE0K0(self):
	exp = self.exp
	params = self.params
        exp.freq = params["freq"]
        exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

        self.checkMatchesParams()

    def testFitFunctionMatchesActualNormalSliceOnlyCapacitanceGivenNoE0K0(self):
	exp = self.exp
	params = self.params
        exp.addDebugParam("No-freq-preprocessing")
        exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])

        self.checkMatchesParams()

    def testFitMatchesActualOverNormalSliceNothingGivenNoE0K0(self):
	exp = self.exp
	params = self.params
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()
