"""Time how long it takes to solve the DE for various values of eq_pot."""

import electrochemistry.tools.fileio as io
import electrochemistry.tools.solution_tools as st
import numpy as np
from profilehooks import profile

PTS_PER_WAVE = 200

@profile
def solve_ode(base_data, time_step, num_time_pts, num_rep=100):
    """Repeatedly solve the reaction ODE for timing purposes."""
    for _ in xrange(num_rep):
        _, _ = st.solve_reaction_from_json(time_step, num_time_pts, base_data)

def main():
    """Rune the timing code."""
    file_name = "./files/simulationParameters.json"
    data_name = "Martin's experiment"
    base_data = io.read_json_params(file_name, data_name)

    end_time = (base_data["pot_rev"] - base_data["pot_start"]) / base_data["nu"]
    num_time_pts = int(np.ceil(PTS_PER_WAVE * base_data["freq"] * 2 * end_time))
    time_step = end_time / (num_time_pts - 1)

    base_data["eq_rate"] = 4e3
    base_data["eq_pot"] = -0.1
    solve_ode(base_data, time_step, num_time_pts)
    base_data["eq_pot"] = -0.2
    solve_ode(base_data, time_step, num_time_pts)
    base_data["eq_pot"] = -0.3
    solve_ode(base_data, time_step, num_time_pts)
    base_data["eq_pot"] = -0.4
    solve_ode(base_data, time_step, num_time_pts)
    base_data["eq_pot"] = -0.5
    solve_ode(base_data, time_step, num_time_pts)
    base_data["eq_pot"] = -0.6
    solve_ode(base_data, time_step, num_time_pts)
    base_data["eq_pot"] = -0.7
    solve_ode(base_data, time_step, num_time_pts)

if __name__ == "__main__":
    main()
