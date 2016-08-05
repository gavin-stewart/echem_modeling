# Tests for the conversion and solutionTools

import unittest
import tools.solutionTools as st
import tools.gridTools as gt
import tools.fileio as io
import tools.conversion as conv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as mpl
import os.path 
import os
from scipy.stats.distributions import norm

class ConversionTests(unittest.TestCase):
         def testTimeConversion(self):
                  """Verify that the time conversion function works"""
                  pot_start = -1
                  pot_end = 1
                  time = np.linspace(0, 1, 100)
                  ramp_rate = 2.
                  temp = 25.+273.15
                  tau = conv.nondim_time(temp, ramp_rate, pot_start, time)
                  self.assertAlmostEqual(tau[0], conv.nondim_pot(temp, pot_start))
                  self.assertAlmostEqual(tau[-1], conv.nondim_pot(temp, pot_end))

class SolutionToolsTests(unittest.TestCase):

         def testMatchesMartinsData(self):
                  """The solve_reaction_dimensional method matches data given 
                  by Martin
                  """
                  fileName = './files/dataMartin.npz'

                  jsonFileName = "./files/simulationParameters.json"
                  dataName = "Martin's experiment"

                  t,IObs =  io.read_time_current_data_bin(fileName)
                  time_step = t[1] - t[0]
                  num_time_pts = len(t)
                  
                  params = io.read_json_dimen_params(jsonFileName, dataName)
         
                  I, amt = st.solve_reaction_from_json(time_step, num_time_pts,
                                                       params)
                  
                  n = len(t)
                  lowerBdd = int(np.ceil(0.2*n))
                  upperBdd = int(np.floor(0.8*n))
                  ind = range(lowerBdd, upperBdd)
                  
                  #Check that model accounts for >= 95% of variation about the mean.
                  self.assertTrue(np.sum((IObs[ind] - I[ind])**2) <=\
                  5e-2 * np.sum((IObs[ind] - np.mean(IObs[ind])**2)))

         def testExtractHarmonicCorrectlyIDsHarmonics(self):
                  """Pure sines match their first harmonic"""
                  x = np.linspace(0,1,1e4)
                  freq = 5
                  y = np.sin(2*np.pi*freq*x)
                  y2 = st.extract_harmonic(1, freq, y)
                  fourier = np.fft.rfft(y)
                  fourier[freq] = 0
                  # since y is a pure tone, it should equal its first harmonic
                  self.assertTrue(np.sum(np.square(y2-y))/len(y) < 1e-5)

         def testExtractHarmonicVanishesForNullHarmonics(self):
                  """Pure functions do not have higher harmonics"""
                  x = np.linspace(0,1,1e4)
                  freq = 5
                  y = np.sin(2*np.pi*freq*x)
                  y2 = st.extract_harmonic(2, freq, y)
                  fourier = np.fft.rfft(y)
                  fourier[freq] = 0
                  errorEst = np.amax(abs(fourier)) * 10/np.sqrt(2)
                  # since y is a pure tone, its second harmonic should vanish
                  self.assertTrue(np.sum(np.square(y2))/len(y) < 1e-5)

         def testExtractHarmonicReturnsRealForRealInput(self):
                  """Real functions have real harmonics"""
                  y = np.random.rand(1e4)
                  y2 = st.extract_harmonic(3, 5, y)
                  # Harmonic should be real for real input data
                  self.assertTrue(np.isclose(np.amax(np.imag(y2)), 0))

         def testSolutionHarmonics1(self):
                  """Second harmonic roughly matches the results of Morris et al 2015"""

                  num_time_pts = 7e4
                  t=np.linspace(0,7,num_time_pts)
                  time_step = t[1] - t[0]
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
         
                  I, amt = st.solve_reaction_dimensional(time_step, num_time_pts, E_0, dE, freq, k_0, 
                  Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, nu, area, 
                  coverage, False)

                  secHarm = st.extract_harmonic(2, freq * 7, I)
                  secHarmMax = np.amax(secHarm)
                  self.assertTrue(secHarmMax > 0.9e-4)
                  self.assertTrue(secHarmMax < 1.3e-4)
         
         def testSolutionHarmonics2(self):
                  """Fifth harmonic roughly matches Morris et al 2015"""
         
                  num_time_pts = int(7e5)
                  t = np.linspace(0,7,num_time_pts)
                  time_step = t[1] - t[0]
                  
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

                  I, amt = st.solve_reaction_dimensional(time_step, num_time_pts, E_0, dE, freq, k_0, 
                  Ru, Cdl, Cdl1, Cdl2, Cdl3, EStart, ERev, temp, nu, area, 
                  coverage, False)


                  secHarm = st.extract_harmonic(5, freq * 7, I)
                  secHarmMax = np.amax(secHarm)
                  self.assertTrue(secHarmMax > 1e-5)
                  self.assertTrue(secHarmMax < 2.e-5)

         def testShortCenteredKaiserWindowFailsForCentersBelowZero(self):
                  hl = 3
                  center = -5
                  N = 10
                  with self.assertRaises(ValueError):
                           st.short_centered_kaiser_window(hl, center, N)

         def testShortCenteredKaiserWindowFailsForCentersAboveN(self):
                  hl = 3
                  center = 11
                  N = 10
                  with self.assertRaises(ValueError):
                           st.short_centered_kaiser_window(hl, center, N)

         def testShortCenteredKaiserWindowRunForLeftBdyBelow0(self):
                  hl = 5
                  center = 4
                  N = 10
                  st.short_centered_kaiser_window(hl,center,N)

         def testShortCenteredKaiserWindowRunsForRightEndpointAboveN(self):
                  hl = 5
                  center = 6
                  N = 10
                  st.short_centered_kaiser_window(hl, center, N)

         def testShortCenteredKaiserWindowVanishesOutisdeSupport(self):
                  self.assertEqual(st.short_centered_kaiser_window(5, 1, 10)[9], 0)
                  self.assertEqual(st.short_centered_kaiser_window(5, 9, 10)[0], 0)

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

         def testPeakExtractionSimple(self):
                  """Test peak extraction on y=x**2."""
                  x = np.linspace(-1,1,1e5+1)
                  y = 1-np.square(x)
                  peakX, peakY = st.extract_peaks(x,y)
                  self.assertEqual(len(peakX), len(peakY))
                  self.assertEqual(len(peakX), 1)
                  self.assertTrue(np.isclose(peakX[0], 0))
                  self.assertTrue(np.isclose(peakY[0], 1))
         
         def testEnvelopeInterpolation(self):
                  t = np.linspace(0, np.pi, 1e5)
                  y = np.sin(t) * np.sin(500*t)
                  yEnvTrue = np.sin(t)
                  yEnvExtracted = st.interpolatedTotalEnvelope(t, y)
                  self.assertEqual(len(yEnvExtracted), len(yEnvTrue))
                  self.assertTrue(np.sum(np.square(yEnvTrue - yEnvExtracted))/len(yEnvTrue) < 1e-4)
