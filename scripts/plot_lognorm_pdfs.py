#!/bin/python
"""Plot log-normal distributions with various coefficents of variation."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def main():
    """Make the plots."""
    coef_vars = [0.5, 1, 2, 4, 8]
    means = [5e2, 1e3, 2e3, 4e3]
    plt.figure(1)
    for coef_var in coef_vars:
        label = "CV: {0}".format(coef_var)
        plot_pdf(2e3, coef_var, label)
    plt.legend()
    plt.title("Log-normal pdfs with various coefficients of variation.")
    plt.xlabel(r"Rate constant ($\mathsf{sec}^{-1}$)")

    plt.figure(2)
    for mean in means:
        label = "mean: {0}".format(mean)
        plot_pdf(mean, 2, label)
    plt.legend()
    plt.title("Log-normal pdfs with various means")
    plt.xlabel(r"Rate constant ($\mathsf{sec}^{-1}$)")
    
    plt.show()

def sigma_from_coef_var(coef_var):
    """Obtain the shape parameter, sigma, from the coefficient of variation.
    """
    return np.sqrt(np.log(1 + np.square(coef_var)))

def plot_pdf(mean, coef_var, label):
    """Plot a single lognormal pdf."""
    sigma = sigma_from_coef_var(coef_var)
    x = np.logspace(-6, 4, 1e4)
    mean_adj = mean * np.exp(sigma ** 2 / 2)
    y = lognorm.pdf(x / mean_adj, sigma) / mean_adj
    plt.plot(x, y, label=label)

if __name__ == "__main__":
    main()

