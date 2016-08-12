#!/bin/python
"""The code for plotting harmonics of current."""
import electrochemistry.tools.solution_tools as st
import numpy as np
import matplotlib.pyplot as plt

def plot_harmonic(harmonic_num, freq, current, pot, trim, color):
    """Plots a given harmonic of the current"""
    harmonic = st.extract_harmonic(harmonic_num, freq, current)
    plt.fill_between(pot[trim:-trim], harmonic[trim:-trim], color=color)

def main():
    """Code to reproduce Fig 6 of Morris et al 2015"""
    num_time_pts = 9.1e5
    time = np.linspace(0, 7, num_time_pts)
    time_step = time[1] - time[0]
    reaction_data = {
        "pot_start" : -0.2, "pot_rev" : 0.5, "eq_pot" : 0,
        "ac_amplitude" : 80e-3, "freq" : 72, "resistance" : 0, "cdl" : 0,
        "cdl1" : 0, "cdl2" : 0, "cdl3" : 0, "temp" : 293, "nu" : 0.1,
        "coverage" : 1e-11, "area" : 1.0
        }
    freq = 72

    pot = np.linspace(-0.2, 0.5, num_time_pts)
    colors = ["blue", "red", "black", "green"]
    rates = [10000, 10000, 1000, 100]

    plot_trim = 1e4  # High-frequency components on the ends make the plot ugly
    for eq_rate, color in zip(rates, colors):
        reaction_data["eq_rate"] = eq_rate
        current, _ = st.solve_reaction_from_json(time_step, num_time_pts,
                                                 reaction_data)
        plot_harmonic(20, freq*7, current, pot, plot_trim, color)
        plt.hold(True)

    plt.show()

if __name__ == "__main__":
    main()

