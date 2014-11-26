# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base processing for HRS data
#from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from astropy import stats
from astropy import modeling as mod


def background(b_arr, niter=3):
    """Determine the background for an array

    Parameters
    ----------
    b_arr: numpy.ndarray
        Array for the determination of the background

    niter: int
        Number of iterations for sigma clipping

    Returns
    -------
    bkgrd: float
        median background value after sigma clipping

    bkstd: float
        Estimated standard deviation based on the median
        absolute deviation

    """
    cl_arr = stats.sigma_clip(b_arr, iters=niter, cenfunc=np.ma.median,
                              varfunc=stats.median_absolute_deviation)
    return np.ma.median(cl_arr), 1.48 * stats.median_absolute_deviation(cl_arr)


def fit_order(data, detect_kernal, xc, order=3, ratio=0.5):
    """Given an array and an overlapping detect_kernal,
    determine two polynomials that would outline the top
    and bottom of the order

    Parameters
    ----------
    data: ~numpy.ndarray
        Image of the orders

    detect_kernal: ~numpy.ndarray
        An array aligned with data that has the approximate
        outline of the order.   The data shoud have a value
        of one for where the order is.

    xc: int
        x-position to determine the width of the order

    order: int
        Order to use for the polynomial fit.

    ratio: float
        Limit at which to determine an order.  It is the
        ratio of the flux in the pixle to the flux at the
        peak.

    Returns
    -------
    y_l: `~astropy.modeling.models.Polynomial1D`
        A polynomial that outlines the bottom of the order

    y_u: `~astropy.modeling.models.Polynomial1D`
        A polynomial that outlines the top of the order

    """

    # first thing to do is determine shape of the order
    sdata = data * detect_kernal
    m_init = mod.models.Polynomial1D(order)
    y, x = np.indices(sdata.shape)
    #g_fit = mod.fitting.LevMarLSQFitter()
    g_fit = mod.fitting.LinearLSQFitter()
    mask = (sdata > 0)
    m = g_fit(m_init, x[mask], y[mask])

    # now the second thing to do is to determine the top and bottom of the
    # order
    y = int(m(xc))
    n_arr = data[:, xc]
    y1 = y - np.where(n_arr[:y] < ratio * data[y, xc])[0].max()
    y2 = np.where(n_arr[y:] < ratio * data[y, xc])[0].min()
    y_u = m.copy()
    y_l = m.copy()

    y_l.parameters[0] = y_l.parameters[0] - y1
    y_u.parameters[0] = y_u.parameters[0] + y2

    return y_l, y_u
