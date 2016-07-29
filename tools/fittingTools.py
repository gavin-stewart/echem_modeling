"""
A module containing functions used to fit parameters to data.
"""

import numpy as np
import cma
import solutionTools as st

class ACExperiment(object):
    """
    A class to contain experimental data.
    """
	
    def __init__(self, IObs, Ru, temp, area, coverage, nu, freq, dE, EStart,
                ERev, tEnd, reverse = True):
        self.IObs = np.ravel(IObs)
	self.Ru = Ru
	self.temp = temp
	self.area = area
	self.coverage = coverage
	self.nu = nu
	self.Cdl = None
	self.Cdl1 = None
	self.Cdl2 = None
	self.Cdl3 = None
	self.freq = freq
	self.dE = dE
	self.EStart = EStart
	self.ERev = ERev
	self.reverse = reverse
	self.t = np.linspace(0, tEnd, len(self.IObs))

    #TODO add a factory method to make an ACExperiment for data contained in 
    #     a JSON file.

    def _capacitanceAndFreqObjectiveFunction(self, capFreqParams):
        Cdl, Cdl1, Cdl2, Cdl3, freq = capFreqParams #Unpack
	# Assume E_0 in the middle of sweep
        E_0 = 0.5 * (self.EStart + self.ERev) 
	k_0 = 0. #No Faradaic current
        # TODO: Implement box constraints to make resampling unneeded.
	try:
            I, _ = st.solveIDimensional(self.t, E_0, self.dE, freq, k_0, 
                             self.Ru, Cdl, Cdl1, Cdl2, Cdl3, self.EStart, 
                             self.ERev, self.temp, self.nu, self.area, 
                             self.coverage, self.reverse)

        except st.ConvergenceError:
            return np.nan
        n = len(self.IObs)
        if self.reverse:
            n_l = n / 20
            n_h =  9 * n / 20
            n_p = n / 2
            return np.sqrt(
                  np.sum(np.square(self.IObs[:n_l] - I[:n_l])) +
                  np.sum(np.square(self.IObs[n_h:n_l+n_p] - I[n_h:n_l+n_p])) +
                  np.sum(np.square(self.IObs[n_h+n_p:] - I[n_h+n_p])))
        else:
            n_l = n / 10
            n_h = 9 * n / 10
            return np.sqrt(np.sum(np.square(self.IObs[:n_l] - I[:n_l])) + 
                           np.sum(np.square(self.IObs[n_h:] - I[n_h:])))

    def fitCapacitanceAndFrequency(self, CdlStart = 0.0, Cdl1Start = 0.0, 
      Cdl2Start = 0.0, Cdl3Start = 0.0):
        """
        Obtains values of the capacitance parameters and frequency using a 
        CMA-ES algorithm.
        """
        start = [CdlStart, Cdl1Start, Cdl2Start, Cdl3Start, self.freq]

	tf = cma.BoxConstraintsLinQuadTransformation([[-5e-4,5e-4], [-5e-3, 5e-3], [-5e-3, 5e-3], [-1e-4, 1e-4], [self.freq * 0.9, self.freq * 1.1]])
        res = cma.fmin(self._capacitanceAndFreqObjectiveFunction, start, 1e-2, {'transformation': [tf.transform, tf.inverse], 'tolfun' : 1e-14, 'tolfunhist' : 1e-14})
        res = res[0]
	self.Cdl = res[0]
	self.Cdl1 = res[1]
	self.Cdl2 = res[2]
	self.Cdl3 = res[3]
	self.freq = res[4]


