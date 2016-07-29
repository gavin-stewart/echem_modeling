import tools.fittingTools as ft
import tools.solutionTools as st
import tools.dataFileIO as dfio
import numpy as np
import unittest
import matplotlib.pyplot as plt

class FittingToolsTestCase(unittest.TestCase):

    def testCapacitanceAndFrequencyFitFunctionMatchesActual(self):
        # Read in data
	fileName = "./files/simulationParameters.json"
	dataName = "Martin's experiment"
	params = dfio.readParametersFromJSON(fileName, dataName)
	tEnd = 2 * (params["ERev"] - params["EStart"]) / params["nu"]
	t = np.linspace(0, tEnd, int(np.ceil(tEnd * params["freq"] * 200)))
	IObs, _ = st.solveIFromJSON(t, params)
	exp = ft.ACExperiment(IObs, params["Ru"], params["temp"], 
                      params["area"], params["coverage"], params["nu"], 
                      params["freq"], params["dE"], params["EStart"], 
                      params["ERev"], tEnd, params["reverse"])
	exp.fitCapacitanceAndFrequency(params["Cdl"], params["Cdl1"], params["Cdl2"], params["Cdl3"])
	
        plt.figure(1)
        plt.plot(exp.t, exp.IObs)

	self.assertAlmostEqual(exp.Cdl, params["Cdl"])
	self.assertAlmostEqual(exp.Cdl1, params["Cdl1"])
	self.assertAlmostEqual(exp.Cdl2, params["Cdl2"])
	self.assertAlmostEqual(exp.Cdl3, params["Cdl3"])
	self.assertAlmostEqual(exp.freq, params["freq"])
	
