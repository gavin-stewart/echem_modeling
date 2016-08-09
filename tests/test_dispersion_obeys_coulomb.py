"""Test that reaction with dispersion proceed to completion and that charge is
conserved.
"""

import electrochemistry.tools.solution_tools as st
import electrochemistry.tools.grid as gt
import electrochemistry.tools.fileio as io
import numpy as np
import unittest

FILE_NAME = './files/simulationParameters.json'

def eq_rate_quad_factory(kSD):
    """Returns a function for constructing quadratures in the rate."""
    return lambda n: gt.hermgauss_param(n, 4e3, kSD, True)

def eq_pot_quad_factory(ESD):
    """Returns a function for constructing quadratures in the potential."""
    return lambda n: gt.hermgauss_param(n, -0.41, ESD, False)

class DispersionCoulombTests(unittest.TestCase): #pylint: disable=too-many-public-methods
    """
    Tests that functions relating to dispersion perform as expected.
    """

    ESDVals = [1e-3, 1e-2, 1e-1]
    kSDVals = [1, 2, 3]
    numSampPts = 15

    def setUp(self): #pylint: disable=invalid-name
        self.baseData = io.read_json_params(FILE_NAME, "disp Coulomb")
        tEnd = (self.baseData["pot_rev"] - self.baseData["pot_start"])\
             / self.baseData["nu"]
        self.num_time_pts = np.ceil(tEnd * 8.959 * 200)

        self.time_step = tEnd / (self.num_time_pts - 1)

    def add_resistance(self):
        """Add the effects of resistance"""
        self.baseData["Ru"] = 100.0

    def add_const_capacitance(self):
        """Add a constant capacitance."""
        self.baseData["Cdl"] = 1e-4

    def add_var_capacitance(self):
        """Add the effects of variable capacitance"""
        self.add_const_capacitance()
        self.baseData["Cdl1"] = 6e-4
        self.baseData["Cdl2"] = 2.5e-4
        self.baseData["Cdl3"] = 1.1e-6

    def add_ac(self):
        """Add an AC component to the voltage"""
        self.baseData["freq"] = 8.959
        self.baseData["ac_amplitude"] = 150e-3

    def check_sat_coulomb(self):
        """Check that charge is conserved.

        This function should only be called after setting up a test.
        """
        I, amtNoDisp = st.solve_reaction_from_json(
            self.time_step, self.num_time_pts, self.baseData)
        endAmt = amtNoDisp[-1]
        INoDispInt = np.sum(I) / self.num_time_pts

        for ESD in self.ESDVals:
            for kSD in self.kSDVals:
                self.baseData["bins"] = gt.product_grid(
                    eq_pot_quad_factory(ESD), self.numSampPts,
                    eq_rate_quad_factory(kSD), self.numSampPts)
                I, amt = st.solve_reaction_from_json(
                    self.time_step, self.num_time_pts, self.baseData)
                self.assertAlmostEqual(endAmt, amt[-1])
                IInt = np.sum(I) / self.num_time_pts
                self.assertAlmostEqual(IInt, INoDispInt)

    def test_dc_no_res_no_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.check_sat_coulomb()


    def test_dc_no_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_resistance()
        self.check_sat_coulomb()

    def test_dc_const_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_resistance()
        self.add_const_capacitance()
        self.check_sat_coulomb()

    def test_dc_var_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_resistance()
        self.add_var_capacitance()
        self.check_sat_coulomb()

    def test_ac_no_res_no_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_ac()
        self.check_sat_coulomb()

    def test_ac_no_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_ac()
        self.add_resistance()
        self.check_sat_coulomb()

    def test_ac_with_const_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_ac()
        self.add_const_capacitance()
        self.add_resistance()
        self.check_sat_coulomb()

    def test_ac_with_var_cap(self):
        """Ensure that the integral of Faradaic current over time is constant.
        """
        self.add_ac()
        self.add_var_capacitance()
        self.add_resistance()
        self.check_sat_coulomb()
