#!/bin/python
"""Plot log-normal distributions with various coefficents of variation."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

MEAN = 2e3

def main():
    """Make the plots."""
    coef_vars = [0.5, 1, 2, 4, 8]
    for coef_var in coef_vars:
        plot_pdf(coef_var)
    plt.legend()
    plt.title("Log-normal pdf for various coefficients of variation.")
    plt.xlabel(r"Rate constant ($\mathsf{sec}^{-1}$)")
    plt.show()

def sigma_from_coef_var(coef_var):
    """Obtain the shape parameter, sigma, from the coefficient of variation.
    """
    return np.sqrt(np.log(1 + np.square(coef_var)))

def plot_pdf(coef_var):
    """Plot a single lognormal pdf."""
    sigma = sigma_from_coef_var(coef_var)
    x = np.logspace(-6, 4, 1e4)
    mean_adj = MEAN * np.exp(sigma ** 2 / 2)
    y = lognorm.pdf(x / mean_adj, sigma) / mean_adj
    label = "CV: {0}".format(coef_var)
    plt.plot(x, y, label=label)

if __name__ == "__main__":
    main()

