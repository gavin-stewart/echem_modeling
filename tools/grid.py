"""A collection of functions for constructing quadrature grids."""
import scipy.stats.distributions
import numpy as np
import itertools

def unif_spaced_param(
        num_samples, start_point, end_point, loc, scale, exp,
        distr=scipy.stats.distributions.norm):
    """Return a grid of points uniformly spaced between start_point and
    end_point, with weights determined by distr.
    """
    if num_samples == 1:
        #Silly, but worth a special case
        return [(start_point + end_point)/2], [1.0]
    #pylint: disable=maybe-no-member
    keys = np.linspace(start_point, end_point, num_samples).tolist()
    #pylint:enable=maybe-no-member
    keys.insert(0, -np.inf)
    keys.append(np.inf)
    if exp:
        pts = loc * np.exp(keys[1:-1]) * exp(-np.square(scale) / 2)
    else:
        pts = keys[1:-1]
    weights = []
    for key, key_prev, key_next in zip(keys[1:-1], keys[:-2], keys[2:]):
        if not exp:
            weights.append(distr.cdf((key + key_next)/2, loc, scale)
                           - distr.cdf((key + key_prev)/2, loc, scale))
        else:
            weights.append(distr.cdf((key + key_next)/2, 0, scale)
                           - distr.cdf((key + key_prev)/2, 0, scale))

        return pts, weights

def unif_spaced_prob(
        num_samples, loc, scale, exp, distr=scipy.stats.distributions.norm):
    """Return a grid of points uniformly spaced in probability space for
    the distribution distr.
    """
    if num_samples == 1:
        #Again, silly, but it must be handled separately.
        return [loc], [1.0]
    bin_ends = np.linspace(0, 1, num_samples + 1)
    bin_centers = 0.5 * (bin_ends[1:] + bin_ends[:-1])
    pts = []
    weights = itertools.repeat(1.0/float(num_samples), num_samples)
    if exp:
        pts = loc * np.exp(distr.ppf(bin_centers, 0, scale))\
            * np.exp(-0.5 * np.square(scale))
    else:
        pts = distr.ppf(bin_centers, loc, scale)
    return pts, weights

def leggauss_param(
        num_samples, start_point, end_point, loc, scale, exp,
        distr=scipy.stats.distributions.norm):
    """Return a grid of points between start_point and end_point chosen
    according to the Gauss-Legendre quadrature rule and weighted according to
    distr.
    """
    gl_pts, gl_weights = np.polynomial.legendre.leggauss(num_samples)
    rescaled_pts = 0.5 * (gl_pts + 1) * (end_point-start_point) + start_point
    if exp:
        pts = loc * np.exp(rescaled_pts)* exp(-0.5 * np.square(scale))
    else:
        pts = rescaled_pts
    weights = []
    for quad_pt, weight in zip(rescaled_pts, gl_weights):
        if exp:
            weights.append(weight * distr.pdf(quad_pt, 0, scale))
        else:
            weights.append(weight * distr.pdf(quad_pt, loc, scale))
    return pts, weights

def leggauss_prob(
        num_samples, loc, scale, exp,
        distr=scipy.stats.distributions.norm):
    """Return a grid of points chosen by applying the Gauss-Legendre quadrature
    rule to [0,1] and applying the inverse cdf transformation of distr.  The
    weights are given according to Gauss-Legendre quadrature theory and the
    distribution pdf.
    """
    gl_pts, gl_weights = np.polynomial.legendre.leggauss(num_samples)
    rescaled = (gl_pts + 1) * 0.5
    if exp:
        log_param_pts = distr.ppf(rescaled, 0, scale)
        pts = loc * np.exp(log_param_pts) * exp(-np.square(scale) / 2)
        weights = np.exp(-np.square(log_param_pts) / (2 * scale ** 2))\
                / (np.sqrt(2*np.pi) * scale) * gl_weights
    else:
        pts = distr.ppf(rescaled, loc, scale)
        weights = np.exp(-np.square(pts - loc) / (2 * scale ** 2))\
                / (np.sqrt(2*np.pi) * scale) * gl_weights
    return pts, weights

def hermgauss_param(num_samples, mean, stdev, exp=False):
    """Computes Gauss-Hermite quadrature points and weights for a given number
    of samples.

    Note that this functions assumes that E_0 and log(k_0) are normally
    distributed.
    """
    if np.isclose(stdev, 0):
        # No variance, will put all the points on mean.
        return [mean], [1]

    gh_pts, weights = np.polynomial.hermite.hermgauss(num_samples)
    if exp:
        pts = mean * np.exp(np.sqrt(2) * stdev * gh_pts)\
            * np.exp(-np.square(stdev) / 2)
    else:
        pts = np.sqrt(2) * stdev * gh_pts + mean
    weights = weights / np.sqrt(np.pi)
    return pts, weights


def product_grid(pot_quad_rule, pot_num_pts, rate_quad_rule, rate_num_pts):
    """Returns the tensor product grid for the given quadrature rules and
    numbers of points.
    """
    return [(pot, rate, pot_weight*rate_weight)
            for pot, pot_weight in zip(*pot_quad_rule(pot_num_pts))
            for rate, rate_weight in zip(*rate_quad_rule(rate_num_pts))]

def sparse_grid(
        pot_quad_rule, rate_quad_rule, level, pot_pts_seq, rate_pts_seq):
    """Returns the sparse grid of a given level for the given quadrature rules.
    """
    if len(pot_pts_seq) < level:
        msg = """The length of EPtsSeq, {0}, was too small for the level {1}\
              """.format(len(pot_pts_seq), level)
        raise ValueError(msg)
    if len(rate_pts_seq) < level:
        msg = """The length of KPtsSeq, {0}, was too small for the level {1}\
              """.format(len(rate_pts_seq), level)
        raise ValueError(msg)

    smolyak_pts = []

    # Smolyak formula for 2D.
    # Positive-weight points
    def quad_concat(num_pts_pot, num_pts_rate):
        """Concatenate the potential and rate quadratures."""
        return pot_quad_rule(num_pts_pot) + rate_quad_rule(num_pts_rate)
    pos_pts_and_weights = map(quad_concat,
                              pot_pts_seq[:level+1],
                              list(reversed(rate_pts_seq[:level+1])))
    for pot_pts, pot_weights, rate_pts, rate_weights in pos_pts_and_weights:
        smolyak_pts.extend(
            [(pot, rate, pot_weight*rate_weight)
             for pot, pot_weight in zip(pot_pts, pot_weights)
             for rate, rate_weight in zip(rate_pts, rate_weights)])

    #Negative-weight points
    neg_pts_and_weights = map(quad_concat,
                              pot_pts_seq[:level],
                              list(reversed(rate_pts_seq[:level])))
    for pot_pts, pot_weights, rate_pts, rate_weights in neg_pts_and_weights:
        smolyak_pts.extend(
            [(pot, rate, -pot_weight*rate_weight)
             for pot, pot_weight in zip(pot_pts, pot_weights)
             for rate, rate_weight in zip(rate_pts, rate_weights)])
    return smolyak_pts
