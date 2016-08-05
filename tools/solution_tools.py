"""A collection of tools for solving for current and manipulating solutions.
"""

import numpy as np
import tools.conversion as conv
import scipy.signal
import scipy.stats.distributions
import scipy.interpolate
from scipy.weave import inline
import copy
from bisect import bisect
from warnings import warn

SOLVER_FUNCTIONS = {}

TOL = float(1e-8)
FTOL = float(2e-6)
MAX_ITER = int(10000)

C_CODE = """// Python expr
int i;
double t = tau_start;
return_val = -1; //All good
Params *parameters = (struct Params*)malloc(sizeof(struct Params));
assert(parameters != NULL);

parameters->rho = rho;
parameters->gamma = gamma;
parameters->gamma1 = gamma1;
parameters->gamma2 = gamma2;
parameters->gamma3 = gamma3;
parameters->rhoDivh = rho/step_size;
parameters->h = step_size;
parameters->kappa = kappa;
parameters->eps_0 = eps_0;


//Set up the I, theta pointers.
parameters->thetaCurr = theta;
parameters->ICurr = current;

if(fabs(parameters->rho) > 1e-10) {
    // Set up the parameter struct.

    for(i = 0; i < num_time_pts - 1; i++) {
        t += step_size;
        double IRadius = (kappa+100) * step_size + 100;

        if(t < tau_rev) {
            parameters->potNext = t + ac_amplitude * sin(omega * (t-tau_start) + phi);
            parameters->dEpsNext = 1 + omega * ac_amplitude * cos(omega * (t-tau_start) + phi);
        } else {
            parameters->potNext = 2 * tau_rev - t + ac_amplitude * sin(omega * (t-tau_start)  + phi);
            parameters->dEpsNext = -1 + omega * ac_amplitude * cos(omega * (t-tau_start) + phi);
        }

        current[i+1] = current[i] + 0.5 * IRadius;
        double currentGuess = current[i] - 0.5 * IRadius;

        if(solverSecantToFP(MAX_ITER, (current+i+1), currentGuess, &discrepParam, TOL, FTOL, parameters) == -1) { //Failed to converge. . .
            return_val = i;
            f_err[0] = discrepParam(current[i+1], parameters);
            break;
        }
        parameters->ICurr++;
        parameters->thetaCurr++;
    }
    free(parameters);
} else {
    double op, dOP;
    for(i = 0; i < num_time_pts - 1; i++) {
        t += step_size;
        if(t < tau_rev) {
            op = t + ac_amplitude * sin(omega * (t-tau_start) + phi) - eps_0;
            dOP = 1 + omega * ac_amplitude * cos(omega * (t-tau_start) + phi);
        } else {
            op = 2 * tau_rev - t + ac_amplitude * sin(omega * (t-tau_start) + phi) - eps_0;
            dOP = -1 + omega * ac_amplitude * cos(omega * (t-tau_start) + phi);
        }
        double expon = exp(0.5 * op);
        double kOx = kappa * expon;
        double kRed = kappa / expon;
        theta[i+1] = (theta[i] + step_size * kOx) / (1 + step_size * (kRed + kOx));
        double dTheta = (1-theta[i+1]) * kOx - theta[i+1] * kRed;
        current[i+1] = gamma * dOP + dTheta;
    }

}"""

OLD_ARGS = ["E_0", "k_0", "dE", "EStart", "ERev"]

def _test_deprecated_keys(kwargs):
    """Test to see if kwargs contains deprecated keys"""
    deprecated_keys = [key for key in OLD_ARGS if key in kwargs]
    for key in deprecated_keys:
        warn("Keyworks contains deprecated key {0}.".format(key))

def solve_reaction_nondimensional(
        tau_start, step_size, num_time_pts, eps_0, ac_amplitude, omega, kappa,
        rho, gamma, gamma1, gamma2, gamma3, tau_rev, kappa_thresh=1e4,
        current_start=0.0, theta_start=0.0, phi=0.0):
    """Solves for current using backwards Euler and returns the result as
    a numpy array.
    """
    # Fixing alpha = 0.5 makes the analytic form, numerical evaluation nicer.
    current = np.empty(num_time_pts)
    theta = np.empty(num_time_pts)
    current[0] = current_start
    theta[0] = theta_start
    phi = float(phi)
    gamma = float(gamma)
    gamma1 = float(gamma1)
    gamma2 = float(gamma2)
    gamma3 = float(gamma3)
    rho = float(rho)
    tau_start = float(tau_start)
    step_size = float(step_size)
    num_time_pts = int(num_time_pts)
    omega = float(omega)
    ac_amplitude = float(ac_amplitude)
    if kappa > kappa_thresh:
        kappa = kappa_thresh
    else:
        kappa = float(kappa)
    tau_rev = float(tau_rev)
    eps_0 = float(eps_0)

    f_err = np.empty(1)

    c_vars = ['num_time_pts', 'rho', 'theta', 'current', 'gamma', 'gamma1',
              'gamma2', 'gamma3', 'tau_start', 'step_size', 'omega',
              'ac_amplitude', 'kappa', 'tau_rev', 'eps_0', 'FTOL', 'TOL',
              'MAX_ITER', 'f_err', 'phi']
    headers = ['"solvers.c"', '"discrepancyFunctions.c"', "<stdlib.h>"]
    inc_dirs = ["/users/gavart/Private/python/electrochemistry/tools"]
    ret_flag = inline(C_CODE, c_vars, headers=headers, include_dirs=inc_dirs)

    if ret_flag != -1:
        msg_unformatted = "Failed to converge at %d.  Values were I[%d]"\
                        + " = %10f, I[%d]=%.10f.  Discrepency was %.10f."\

        msg = msg_unformatted %(ret_flag+1, ret_flag, current[ret_flag],
                                ret_flag+1, current[ret_flag+1], f_err[0])
        raise ConvergenceError(msg)
    return current, theta


SOLVER_FUNCTIONS["nondimensional"] = solve_reaction_nondimensional

def solve_reaction_from_json(time_step, num_time_pts, json_data):
    """Solves for the current and amount of species A using backwards Euler
    and returns the results as numpy arrays.
    """
    json_data = copy.copy(json_data)
    json_data.pop('name', None)

    try:
        type_val = json_data.pop('type')
    except KeyError:
        raise ValueError("Expected json_data to include a key 'type'")

    if type_val not in  SOLVER_FUNCTIONS:
        raise ValueError("Unknown type {0}".format(type_val))
    else:
        solver = SOLVER_FUNCTIONS[type_val]
        return solver(time_step, num_time_pts, **json_data)


def solve_reaction_dimensional(
        time_step, num_time_pts, eq_pot, ac_amplitude, freq, eq_rate,
        resistance, cdl, cdl1, cdl2, cdl3, pot_start, pot_rev, temp, nu, area,
        coverage, phi=0.0, **kwargs):
    """Solves for current using backwards Euler and returns the result as a
    numpy array.

    This method is a wrapper around the nondimensional method.
    """
    _test_deprecated_keys(kwargs)
    eps_start = conv.nondim_pot(temp, pot_start)
    eps_rev = conv.nondim_pot(temp, pot_rev)
    ac_amplitude = conv.nondim_pot(temp, ac_amplitude)
    eps_0 = conv.nondim_pot(temp, eq_pot)

    omega = conv.freq_to_dim_omega(temp, nu, freq)

    kappa = conv.nondim_rate(temp, nu, eq_rate)
    kappa_thresh = conv.nondim_rate(temp, nu, 1e6)

    time_step /= conv.scale_time(temp, nu)

    rho = conv.nondim_resistance(temp, nu, area, coverage, resistance)

    # Note: the parameters cdl1...3 are already nondimensional and should
    # NOT be multiplied by the scale potential E0
    gamma = conv.nondim_capacitance(temp, nu, area, coverage, cdl)
    gamma1 = cdl1
    gamma2 = cdl2
    gamma3 = cdl3

    current_nondim, theta = solve_reaction_nondimensional(
        eps_start, time_step, num_time_pts, eps_0, ac_amplitude, omega,
        kappa, rho, gamma, gamma1, gamma2, gamma3, eps_rev,
        kappa_thresh=kappa_thresh, phi=phi)

    current_dimen = conv.dim_current(temp, nu, area, coverage, current_nondim)
    amt_covered = theta * area * coverage

    return current_dimen, amt_covered
SOLVER_FUNCTIONS["dimensional"] = solve_reaction_dimensional


def solve_reaction_disp_dim_bins(
        time_step, num_time_pts, ac_amplitude, freq, resistance, cdl, cdl1,
        cdl2, cdl3, pot_start, pot_rev, temp, nu, area, coverage, bins,
        **kwargs):
    """Solves for current when there is dispersion.

    The variable bins should be a list of tuples of the form
    (E_0Val, K_0Val, weight).
    """
    _test_deprecated_keys(kwargs)
    current_avg = np.zeros(num_time_pts)
    amt_avg = np.zeros(num_time_pts)
    for vals in bins:
        eq_potential = vals[0]
        rate_const = vals[1]
        weight = vals[2]

        current, amt = solve_reaction_dimensional(time_step, num_time_pts,
                                                  eq_potential, ac_amplitude,
                                                  freq, rate_const, resistance,
                                                  cdl, cdl1, cdl2, cdl3,
                                                  pot_start, pot_rev, temp, nu,
                                                  area, coverage)
        current_avg += weight * current
        amt_avg += weight * amt
    return current_avg, amt_avg
SOLVER_FUNCTIONS["disp-dimensional-bins"] = solve_reaction_disp_dim_bins

def half_maximum_width(current, time, nu, t_rev_index=None):
    """Returns the width at half-maximum for a current I driven by a dc
    current with ramp nu.
    """
    if t_rev_index is None:
        t_rev_ind = len(time)-1
    current = current[:t_rev_ind]
    max_current = -np.inf
    max_current_ind = None
    for ind, current_val in enumerate(current):
        if current_val > max_current:
            max_current = current_val
            max_current_ind = ind

    half_max = max_current * 0.5
    left = bisect(current, half_max, 0, max_current_ind)
    # Reversed bisection search
    right = t_rev_ind - bisect(list(reversed(current)), half_max,
                               0, t_rev_ind- max_current_ind)
    return (time[right] - time[left]) * nu

def short_centered_kaiser_window(halflength, center, array_size):
    """Returns a Kaiser window of given half-length and center."""
    window = np.zeros(array_size)
    if center < 0 or center > array_size:
        msg_unformatted = "Center {0} is outside the range of values [0,{1}]"
        msg = msg_unformatted.format(center, array_size)
        raise ValueError(msg)
    window_left = int(np.floor(center - halflength))
    window_right = int(np.ceil(center + halflength))

    if window_left >= 0:
        window_left_trunc = window_left
        left_trunc = 0
    else:
        window_left_trunc = 0
        left_trunc = window_left_trunc - window_left

    if window_right < array_size:
        window_right_trunc = window_right
        right_trunc = left_trunc + window_right_trunc - window_left_trunc + 1
    else:
        window_right_trunc = array_size - 1
        right_trunc = left_trunc + window_right_trunc - window_left_trunc + 1
    kaiser = scipy.signal.kaiser(window_right-window_left+1, 14)
    kaiser = kaiser[left_trunc:right_trunc]
    window[window_left_trunc:window_right_trunc+1] = kaiser
    return window

def extract_harmonic(harmonic_number, freq, data):
    """Returns the nth harmonic of a given frequency extracted from real data.

    Note that a Kaiser window is used to extract the harmonic."""
    fourier = np.fft.rfft(data)
    window_halfwidth = 0.75 * freq
    window = short_centered_kaiser_window(window_halfwidth,
                                          harmonic_number*freq, len(fourier))
    return np.fft.irfft(fourier * window)

def solve_reaction_disp_dim_mc(
        time_step, num_time_pts, num_runs, ac_amplitude, freq, resistance,
        cdl, cdl1, cdl2, cdl3, pot_start, pot_rev, temp, nu, area, coverage,
        eq_pot_mean, eq_pot_stdev, eq_rate_mean, eq_rate_stdev, **kwargs):
    """Solves the reaction with dispersion using Monte-Carlo sampling."""
    _test_deprecated_keys(kwargs)
    if eq_pot_stdev == 0:
        eq_pot_vals = np.repeat(eq_pot_mean, num_runs)
    else:
        eq_pot_vals = np.random.normal(eq_pot_mean, eq_pot_stdev, num_runs)
    if eq_rate_stdev == 0:
        eq_rate_vals = np.repeat(eq_rate_mean, num_runs)
    else:
        eq_rate_vals = eq_rate_mean\
                     * np.power(2,
                                np.random.normal(0, eq_rate_stdev, num_runs))

    def solve_reaction_inner(eq_pot, eq_rate):
        """Wrapper around solve_reaction_dimensional."""
        return solve_reaction_dimensional(time_step, num_time_pts, eq_pot,
                                          ac_amplitude, freq, eq_rate,
                                          resistance, cdl, cdl1, cdl2, cdl3,
                                          pot_start, pot_rev, temp, nu, area,
                                          coverage)
    solution_list = [solve_reaction_inner(eq_pot, eq_rate)
                     for eq_pot, eq_rate in eq_pot_vals, eq_rate_vals]
    current_agg, amt_agg = reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]),
                                  solution_list)
    return current_agg / num_runs, amt_agg / num_runs
SOLVER_FUNCTIONS["disp-dimensional-MC"] = solve_reaction_disp_dim_mc

def extract_peaks(x_vals, y_vals, passes=3):
    """Returns a tuple containing a list of x values and yvalues where the
    maximums of the data x,y occur.
    """
    if len(x_vals) != len(y_vals):
        raise ValueError('Expected x and y to have equal lengths')
    #Sort the x values.
    zipped = zip(x_vals, y_vals)
    zipped.sort(key=lambda z: z[0]) #Sort by x value.
    x_vals, y_vals = map(list, zip(*zipped))
    # Dummy values to allow the endpoints to be peaks
    x_vals.insert(0, -np.inf)
    y_vals.insert(0, -np.inf)
    x_vals.append(np.inf)
    y_vals.append(-np.inf)

    x_peaks, y_peaks = map(list, zip(*[
        (x, y) for x, y, y_prev, y_next
        in zip(x_vals[1:-1], y_vals[1:-1], y_vals[:-2], y_vals[2:])
        if y > y_prev and y > y_next
        ]))

    # Go through the list a few times (3, by default), and remove everything
    # that doesn't look like a peak.
    if passes > len(x_peaks):
        passes = len(x_peaks) - 1
    for _ in xrange(1, passes):
        x_peaks, y_peaks = map(list, zip(*[
            (x, y) for x, y, y_prev, y_next
            in zip(x_peaks[1:-1], y_peaks[1:-1], y_peaks[:-2], y_peaks[2:])
            if y >= y_prev or y >= y_next
            ]))
    return x_peaks, y_peaks

def interpolated_total_envelope(x_vals, y_vals):
    """Returns an estimate of the envelope of the signal (x,y) by
    interpolating the maxima of (x,abs(y))
    """
    x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals), key=lambda n: n[0]))
    x_peaks, y_peaks = extract_peaks(x_vals, abs(y_vals))
    if x_vals[0] != x_peaks[0]:
        x_peaks.insert(0, x_vals[0])
        y_peaks.insert(0, abs(y_vals[0]))
    if x_vals[-1] != x_peaks[-1]:
        x_peaks.append(x_vals[-1])
        y_peaks.append(abs(y_vals[-1]))
    return scipy.interpolate.interp1d(np.array(x_peaks), np.array(y_peaks),
                                      kind='linear')(x_vals)

def interpolated_upper_envelope(x_vals, y_vals):
    """Returns an estimate of the envelope of the signal (x,y) by
    interpolating the maxima of (x, y)
    """
    x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals), key=lambda n: n[0]))
    x_peaks, y_peaks = extract_peaks(x_vals, y_vals)
    if x_vals[0] != x_peaks[0]:
        x_peaks.insert(0, x_vals[0])
        y_peaks.insert(0, abs(y_vals[0]))
    if x_vals[-1] != x_peaks[-1]:
        x_peaks.append(x_vals[-1])
        y_peaks.append(abs(y_vals[-1]))
    return scipy.interpolate.interp1d(np.array(x_peaks), np.array(y_peaks),
                                      kind='linear')(x_vals)

def interpolated_lower_envelope(x_vals, y_vals):
    """Returns an estimate of the envelope of the signal (x,hy) by
    interpolating the maxima of (x, -y)
    """
    x_vals, y_vals = map(list, zip(*sorted(zip(x_vals, y_vals),
                                           key=lambda n: n[0])))
    x_peaks, y_peaks = extract_peaks(x_vals, -y_vals)
    if x_vals[0] != x_peaks[0]:
        x_peaks.insert(0, x_vals[0])
        y_peaks.insert(0, abs(y_vals[0]))
    if x_vals[-1] not in x_peaks[-1]:
        x_peaks.append(x_vals[-1])
        y_peaks.append(abs(y_vals[-1]))
    #Sort by x value
    return -1 * scipy.interpolate.interp1d(np.array(x_peaks),
                                           np.array(y_peaks),
                                           kind='linear')(x_vals)

def solver_names():
    """Return the names of the solver functions contained in this module."""
    return SOLVER_FUNCTIONS.keys()

class ConvergenceError(RuntimeError):
    """An error indicating a numerical method has failed to converge."""
    def __init__(self, *args):
        super(ConvergenceError, self).__init__(*args)
