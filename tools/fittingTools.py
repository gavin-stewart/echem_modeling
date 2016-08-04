"""
A module containing functions used to fit parameters to data.
"""

import numpy as np
import cma
import solutionTools as st
import matplotlib.pyplot as plt

class ACExperiment(object):
    """A class to contain experimental data.
    """

    #TODO: Make these object level?
    CdlRescale = 1e4
    Cdl1Rescale = 1.67e3
    Cdl2Rescale = 5e3
    Cdl3Rescale = 1e6
    freqRescale = 2e2
    k_0Rescale = 0.1

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
        self.dE = dE
        self.EStart = EStart
        self.ERev = ERev
        self.reverse = reverse
        self.t = np.linspace(0, tEnd, len(self.IObs))
        self.E_0 = 0.5 * (EStart + ERev)
        self.k_0 = 100.0
        self.E_0Rescale = 2.0 / (ERev - EStart) 
         self.debugParams = []
        n = len(IObs)
        self._freqPreprocessing()
        self.noFaradaicSlices = []
         if reverse:
           self.noFaradaicSlices.append(np.s_[:n / 10])
           self.noFaradaicSlices.append(np.s_[4 * n / 10: 6 * n / 10])
           self.noFaradaicSlices.append(np.s_[9 * n / 10:])
        else:
           self.noFaradaicSlices.append(np.s_[:n / 5])
           self.noFaradaicSlices.append(np.s_[4 * n / 5:])

    def addDebugParam(self, param):
        """add a parameter to be used for debugging"""
        self.debugParams.append(param)

    #TODO add a factory method to make an ACExperiment from data contained in 
    #     a JSON file.

    def _capacitanceAndFreqObjectiveFunction(self, capFreqParams):
        Cdl, Cdl1, Cdl2, Cdl3, freq = self._unscaleCapFreqParams(capFreqParams)
        try:
            I, _ = st.solve_reaction_dimensional(self.t, self.E_0, self.dE,
                                                 freq, self.k_0, self.Ru, Cdl,
                                                 Cdl1, Cdl2, Cdl3,
                                                 self.EStart, self.ERev, 
                                                 self.temp, self.nu, self.area,
                                                 self.coverage)

        except st.ConvergenceError:
            return np.nan
        retSum = 0.0
        for s in self.noFaradaicSlices:
            retSum += np.sum(np.square(self.IObs[s] - I[s]))
        return retSum

    def _timeDomainObjectiveFunction(self, params):
        Cdl, Cdl1, Cdl2, Cdl3, freq, E_0, k_0 = self._unscaleAllParams(params)
        try:
            I, _ = st.solve_reaction_dimensional(self.t, E_0, self.dE, freq, 
                                                 k_0, self.Ru, Cdl, Cdl1, Cdl2,
                                                 Cdl3, self.EStart, self.ERev,
                                                 self.temp, self.nu,
                                                 self.area, self.coverage)

        except st.ConvergenceError:
            return np.nan
         retSum = 0.0
         for s in self.noFaradaicSlices:
            retSum += np.sum(np.square(self.IObs[s] - I[s]))
         return retSum

    def _rescaleCapFreqParams(self, paramList):
        Cdl, Cdl1, Cdl2, Cdl3, freq = paramList
        return [Cdl * self.CdlRescale,
                Cdl1 * self.Cdl1Rescale,
                Cdl2 * self.Cdl2Rescale,
                Cdl3 * self.Cdl3Rescale,
                (freq - self.freq) * self.freqRescale]

    def _unscaleCapFreqParams(self, paramList):
        (CdlRescaled, Cdl1Rescaled, Cdl2Rescaled, 
                        Cdl3Rescaled, freqRescaled) = paramList
        return [CdlRescaled / self.CdlRescale,
                Cdl1Rescaled / self.Cdl1Rescale,
                Cdl2Rescaled / self.Cdl2Rescale,
                Cdl3Rescaled / self.Cdl3Rescale,
                self.freq + freqRescaled / self.freqRescale]

    def _rescaleAllParams(self, paramList):
        Cdl, Cdl1, Cdl2, Cdl3, freq, E_0, k_0 = paramList
        CdlRescaled, Cdl1Rescaled, Cdl2Rescaled, Cdl3Rescaled, freqRescaled =\
                      self._rescaleCapFreqParams([Cdl, Cdl1, Cdl2, Cdl3, freq])
        return [CdlRescaled, Cdl1Rescaled, Cdl2Rescaled, Cdl3Rescaled, 
                freqRescaled, E_0 * self.E_0Rescale,
                np.log(k_0) * self.k_0Rescale]

    def _unscaleAllParams(self, paramList):
        (CdlRescaled, Cdl1Rescaled, Cdl2Rescaled, Cdl3Rescaled, 
                        freqRescaled, E_0Rescaled, k_0Rescaled) = paramList
        Cdl, Cdl1, Cdl2, Cdl3, freq = self._unscaleCapFreqParams([CdlRescaled,
                     Cdl1Rescaled, Cdl2Rescaled, Cdl3Rescaled, freqRescaled])
        return [Cdl, Cdl1, Cdl2, Cdl3, freq, E_0Rescaled / self.E_0Rescale,
                np.exp(k_0Rescaled / self.k_0Rescale)]

    def _freqPreprocessing(self):
        """Approximate the frequency by taking the average distance between 
        the peaks in IObs.
        """
        tPeak, _ = st.extract_peaks(self.t, self.IObs)
        if len(tPeak) < 2:
            pass
        else:
            self.freq = 1.0 / np.mean(np.diff(tPeak))

    def plotFitDiscrepancy(self):
        """Plot the discrepency between the fitted values and observations.
        """
        IFit, _ = st.solve_reaction_dimensional(self.t, self.E_0, self.dE,
                                                self.freq, self.k_0, self.Ru,
                                                self.Cdl, self.Cdl1, self.Cdl2,
                                                self.Cdl3, self.EStart,
                                                self.ERev, self.temp, self.nu,
                                                self.area, self.coverage)

        plt.plot(self.t, abs(self.IObs - IFit))

    def fitCapacitanceAndFrequency(self, CdlStart = 1e-4, Cdl1Start = 0.0, 
      Cdl2Start = 0.0, Cdl3Start = 0.0):
        """Obtains values of the capacitance parameters and frequency using a 
        CMA-ES algorithm.
        """
        start = [CdlStart, Cdl1Start, Cdl2Start, Cdl3Start, self.freq]
        start = self._rescaleCapFreqParams(start)
         verbose = 1 if "loud-cma" in self.debugParams else -9
        optDict = {'tolfun' : 1e-19, 'tolfunhist' : 1e-19, "verbose" : verbose,
                   'verb_log' : np.inf}
        res = cma.fmin(self._capacitanceAndFreqObjectiveFunction, start, 1, 
                       optDict) 
        res = self._unscaleCapFreqParams(res[0])
         print res
         self.Cdl = res[0]
         self.Cdl1 = res[1]
         self.Cdl2 = res[2]
         self.Cdl3 = res[3]
         self.freq = res[4]

    def fitAllParamsTimeDomain(self, CdlStart = 1e-4, Cdl1Start = 0.0, 
                         Cdl2Start = 0.0, Cdl3Start = 0.0):
        """Obtains the values of the frequency, capacitance, rate and 
        thermodynamic parameters using a CMA-ES algorithm. 
        """
        start = [CdlStart, Cdl1Start, Cdl2Start, Cdl3Start, self.freq, 
                  self.E_0, self.k_0]
        start = self._rescaleAllParams(start)
        verbose = 1 if "loud-cma" in self.debugParams else -9
        optDict = {'tolfun' : 1e-14, 'tolfunhist' : 1e-15, "verbose" : verbose,
                   'verb_log' : np.inf}
        res = cma.fmin(self._timeDomainObjectiveFunction, start, 1, 
                       optDict) 
        res = self._unscaleAllParams(res[0])
        print res
        self.Cdl = res[0]
        self.Cdl1 = res[1]
        self.Cdl2 = res[2]
        self.Cdl3 = res[3]
        self.freq = res[4]
        self.E_0 = res[5]
        self.k_0 = res[6]




