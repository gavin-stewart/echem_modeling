import electrochemistry.tools.fittingTools as ft
import electrochemistry.tools.solution_tools as st
import electrochemistry.tools.fileio as io
import numpy as np
import unittest

class FittingToolsCapacitanceAndFrequencyTestCase(unittest.TestCase):
    """"
    Give the fitting procedure the true parameters as a starting point.
    """

    @classmethod
    def setUpClass(cls):
        # Read in data
         fileName = "./files/simulationParameters.json"
         dataName = "Martin's experiment"
         cls.params = io.read_json_params(fileName, dataName)
         cls.tEnd = 2 * (cls.params["ERev"] - cls.params["EStart"])\
                  / cls.params["nu"]
         cls.num_time_pts = int(np.ceil(cls.tEnd * cls.params["freq"] * 200))
         cls.time_step = cls.tEnd / cls.num_time_pts
         cls.IObs, _ = st.solve_reaction_from_json(cls.time_step,
                                                   cls.num_time_pts,
                                                   cls.params)


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
        self.exp.E_0 = self.params["eq_pot"]
        self.exp.k_0 = self.params["eq_rate"]

    def setE0ToCorrect(self):
        self.exp.E_0 = self.params["eq_pot"]

    def setk0ToCorrect(self):
        self.exp.k_0 = self.params["eq_rate"]

    def setSliceToWholeTime(self):
        self.exp.noFaradaicSlices = [np.s_[:]]


    def checkMatchesParams(self):
        exp = self.exp
        self.assertAlmostEqual(exp.Cdl, self.params["cdl"], 6)
        self.assertAlmostEqual(exp.Cdl1, self.params["cdl1"], 6)
        self.assertAlmostEqual(exp.Cdl2, self.params["cdl2"], 6)
        self.assertAlmostEqual(exp.Cdl3, self.params["cdl3"], 7)
        self.assertAlmostEqual(exp.freq, self.params["freq"], 3)

    @unittest.skip("No whole time")
    def testFitFunctionMatchesActualObjectiveOverWholeTimeOnlyCapacitanceGiven(
            self):
        exp = self.exp
        self.setSliceToWholeTime()
        self.setE0k0ToCorrect()
        exp.fitCapacitanceAndFrequency(self.params["cdl"], self.params["cdl1"],
                                       self.params["cdl2"],
                                       self.params["cdl3"])

        self.checkMatchesParams()

    @unittest.skip("No whole time")
    def testFitMatchesActualOverWholeTimeNothingGiven(self):
        exp = self.exp
        self.setSliceToWholeTime()
        self.setE0k0ToCorrect()
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()

    def testFitMatchesActualOverNormalSliceNothingGiven(self):
        exp = self.exp
        self.setE0k0ToCorrect()
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()

    def testFitFunctionMatchesActualNormalSliceOnlyCapacitanceGivenNoE0(self):
        exp = self.exp
        self.setk0ToCorrect()
        exp.fitCapacitanceAndFrequency(self.params["cdl"],
                                       self.params["cdl1"],
                                       self.params["cdl2"],
                                       self.params["cdl3"])

        self.checkMatchesParams()

    def testFitMatchesActualOverNormalSliceNothingGivenNoE0(self):
        exp = self.exp
        self.setk0ToCorrect()
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()

    def testFitFunctionMatchesActualNormalSliceOnlyCapacitanceGivenNoE0k0(self):
        exp = self.exp
        exp.fitCapacitanceAndFrequency(self.params["cdl"],
                                       self.params["cdl1"],
                                       self.params["cdl2"],
                                       self.params["cdl3"])

        self.checkMatchesParams()

    def testFitMatchesActualOverNormalSliceNothingGivenNoE0k0(self):
        exp = self.exp
        exp.fitCapacitanceAndFrequency()

        self.checkMatchesParams()
