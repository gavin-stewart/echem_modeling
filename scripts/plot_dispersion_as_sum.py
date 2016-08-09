"""Plot the current due to dispersion, and represent it as the weighted sum of
5 components.
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import electrochemistry.tools.solution_tools as st
import electrochemistry.tools.grid as gt
import electrochemistry.tools.fileio as io
import numpy as np

def plot_envelopes(time, current, color):
    """Plot the upper and lower envelopes of a current."""
    env_upper = st.interpolated_upper_envelope(time, current)
    env_lower = st.interpolated_lower_envelope(time, current)
    plt.plot(time, env_upper, color=color)
    plt.plot(time, env_lower, color=color)


def main():
    """Make the dispersion plot."""
    eq_pot_stdev = 0.05
    file_name = 'files/simulationParameters.json'
    data = io.read_json_params(file_name, "Martin's experiment")
    time_end = 2 * (data["pot_rev"] - data["pot_start"]) / data["nu"]
    num_time_pts = int(np.ceil(200 * data["freq"] * time_end))
    time_step = time_end / (num_time_pts - 1)
    time = np.linspace(0, time_end, num_time_pts)

    eq_pot_grid = gt.hermgauss_param(5, data["eq_pot"], eq_pot_stdev, False)

    color_scale = eq_pot_grid[0]
    # Rescale to [0,1]
    color_scale = (color_scale - np.min(color_scale))\
               / (np.max(color_scale) - np.min(color_scale))

    colors = [cm.bwr(x) for x in color_scale] #pylint: disable=no-member

    #Generate currents
    # Start with capacitive current
    eq_rate = data["eq_rate"]
    data["eq_rate"] = 0
    cap_current, _ = st.solve_reaction_from_json(time_step, num_time_pts, data)
    data["eq_rate"] = eq_rate

    disp_current = np.zeros(num_time_pts)

    # Now, compute the component currents.
    for color, eq_pot, weight in zip(colors, *eq_pot_grid):
        data["eq_pot"] = eq_pot
        current, _ = st.solve_reaction_from_json(time_step, num_time_pts, data)
        current -= cap_current
        current *= weight
        plot_envelopes(time, current, color)
        disp_current += current

    plt.fill_between(time, disp_current, -disp_current)
    plt.show(block=True)

if __name__ == "__main__":
    main()

