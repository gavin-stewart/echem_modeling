#!/bin/python
"""Plot and display the harmonics for the dispersion data."""

import matplotlib.pyplot as plt
import electrochemistry.tools.fileio as io
import electrochemistry.tools.solution_tools as st
import electrochemistry.tools.grid as gt
import scipy.signal
import numpy as np
import matplotlib.collections
import itertools
import matplotlib.cm as cm
import os.path

TRIM = slice(5000, -5000)
COLORS = itertools.cycle(
    cm.rainbow(np.linspace(0, 1, 4))) #pylint: disable=no-member

NUM_EVALS = 30

FILE_NAME = io.get_file_resource_path("simulationParameters.json")

I_ENV_LIM = 0


EQ_POT_STDEV = [1e-3, 1e-2, 1e-1]
EQ_RATE_STDEV = [1, 2, 3]

def main():
    current_env_lim = 0

    base_data = io.read_json_params(FILE_NAME, "Martin's experiment")
    freq = base_data["freq"]
    eq_pot_bin_fact = lambda stdev: lambda n: gt.hermgauss_param(
        n, base_data["eq_pot"], stdev, False)
    eq_rate_bin_fact = lambda stdev: lambda n: gt.hermgauss_param(
        n, base_data["eq_rate"], stdev, True)

    plotsE0 = plt.figure(1)
    plotsk0 = plt.figure(2)
    plotsboth = plt.figure(3)

    plotsIE0 = plotsE0.add_subplot(211)
    plotsHarmE0 = plotsE0.add_subplot(212)
    plotsIk0 = plotsk0.add_subplot(211)
    plotsHarmk0 = plotsk0.add_subplot(212)
    plotsIboth = plotsboth.add_subplot(211)
    plotsHarmboth = plotsboth.add_subplot(212)

    def plotCurrentAndHarmonics(current, label, currentAx, harmAx):
        """Plot the upper and lower envelopes of a current"""
        global I_ENV_LIM
        plotColor = next(COLORS)
        IEnvUpper = st.interpolated_upper_envelope(time, current-ICap)
        IEnvLower = st.interpolated_lower_envelope(time, current-ICap)
        IEnvMax = max(np.amax(abs(IEnvUpper)), np.amax(abs(IEnvLower)))
        if IEnvMax > I_ENV_LIM:
            I_ENV_LIM = IEnvMax
        envelopes = matplotlib.collections.LineCollection(
            [list(zip(time, IEnvUpper)), list(zip(time, IEnvLower))],
            label=label,
            color=plotColor)
        currentAx.add_collection(envelopes)
        harm10 = st.extract_harmonic(10, freq * time[-1], current)[TRIM]
        harm10 = abs(scipy.signal.hilbert(harm10))
        harmAx.plot(harm10, label=label, color=plotColor)

    def setupAndSavePlots(fig, currentAx, harmAx, name):
        """Creates the plots and write them to a file."""
        leg = harmAx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),
                            mode="expand", borderaxespad=0., ncol=2)
        currentAx.set_xlim([time[0], time[-1]])
        currentAx.set_ylim([-1.25 * I_ENV_LIM, 1.25 * I_ENV_LIM])
        fig.savefig(os.path.join(figLoc, name), bbox_extra_artists=(leg,),
                    bbox_inches='tight', pad_inches=0.5)



    # Read in where the plots should be saved
    figLoc = raw_input(
        "Enter the directory where the figures should be saved: ")

# No dispersion.
    tEnd = 2 * (base_data["pot_rev"] - base_data["pot_start"]) / base_data["nu"]
    num_time_pts = int(np.ceil(tEnd * base_data["freq"] * 200))
    time_step = tEnd / num_time_pts
    time = np.linspace(0, tEnd, num_time_pts)
    INoDisp, _ = st.solve_reaction_dimensional(
        time_step, num_time_pts, **base_data)
#Generate a zero-capacitance model
    eq_rate_tmp = base_data["eq_rate"]
    base_data["eq_rate"] = 0
    ICap, _ = st.solve_reaction_dimensional(
        time_step, num_time_pts, **base_data)
    base_data["eq_rate"] = eq_rate_tmp
    base_data["type"] = "disp-dimensional-bins"


# Dispersion in E_0 only.
    eq_rate_bins = eq_rate_bin_fact(0.0)
    plotCurrentAndHarmonics(INoDisp, "No disp", plotsIE0, plotsHarmE0)
    for eq_pot_stdev in EQ_POT_STDEV:
        eq_pot_bins = eq_pot_bin_fact(eq_pot_stdev)
        base_data["bins"] = gt.product_grid(eq_pot_bins, NUM_EVALS,
                                            eq_rate_bins, 1)
        current, _ = st.solve_reaction_from_json(time_step, num_time_pts, base_data)
        label = "E_0 disp " + str(eq_pot_stdev)
        plotCurrentAndHarmonics(current, label, plotsIE0, plotsHarmE0)



# Dispersion in k_0 only.
    eq_pot_bins = eq_pot_bin_fact(0.0)
    plotCurrentAndHarmonics(INoDisp, "No disp", plotsIk0, plotsHarmk0)
    for eq_rate_stdev in EQ_RATE_STDEV:
        eq_rate_bins = eq_rate_bin_fact(eq_rate_stdev)
        base_data["bins"] = gt.product_grid(eq_pot_bins, 1,
                                            eq_rate_bins, NUM_EVALS)
        current, _ = st.solve_reaction_from_json(time_step, num_time_pts, base_data)
        label = "k_0 disp " + str(eq_rate_stdev)
        plotCurrentAndHarmonics(current, label, plotsIk0, plotsHarmk0)



# Dispersion in both E_0 and k_0.
    plotCurrentAndHarmonics(INoDisp, "No disp", plotsIboth, plotsHarmboth)
    for eq_pot_stdev, eq_rate_stdev in zip(EQ_POT_STDEV, EQ_RATE_STDEV):
        eq_pot_bins = eq_pot_bin_fact(eq_pot_stdev)
        eq_rate_bins = eq_rate_bin_fact(eq_rate_stdev)
        base_data["bins"] = gt.product_grid(eq_pot_bins, NUM_EVALS,
                                            eq_rate_bins, NUM_EVALS)
        current, _ = st.solve_reaction_from_json(time_step, num_time_pts, base_data)
        label = "E_0 disp " + str(eq_pot_stdev) + " k_0 disp "\
              + str(eq_rate_stdev)
        plotCurrentAndHarmonics(current, label, plotsIboth, plotsHarmboth)

#Save plots
    setupAndSavePlots(plotsE0, plotsIE0, plotsHarmE0, "E0Dispersion.pdf")

    print "Saved E0 dispersion plots"

    setupAndSavePlots(plotsk0, plotsIk0, plotsHarmk0, "k0Dispersion.pdf")
    print "Saved k0 dispersion plots"

    setupAndSavePlots(plotsboth, plotsIboth,
                      plotsHarmboth, "bothDispersion.pdf")
    print "Saved both dispersion plots"

if __name__ == "__main__":
    main()
