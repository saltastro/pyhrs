# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the tools used to process HRS data
#from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from astropy import stats
from astropy import modeling as mod

__all__ = ['background', 'fit_order', 'normalize_image', 'fit_wavelength_solution']

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

def normalize_image(data, func_init, mask,
                    fitter=mod.fitting.LinearLSQFitter,
                    normalize=True):
    """Normalize an HRS image.

    The tasks takes an image and will fit a function to the overall
    shape to it.  The task will only fit to the illuminated orders
    and if an order_frame is provided it will use that to identify the
    areas it should fit to.  Otherwise, it will filter the image such that
    only the maximum areas are fit.

    This function will then be divided out of the image and return
    a normalized image if requested.

    Parameters
    ----------
    data: numpy.ndarray
         Data to be normalized

    mask: numpy.ndarray
         If a numpy.ndarray, this will be used to determine areas
         to be used for the fit.

    func_init: ~astropy.modeling.models
         Function to fit to the image

    fitter: ~astropy.modeling.fitting
         Fitter function

    normalize: boolean
         If normalize is True, it will return data normalized by the
         function fit to it.  If normalize is False, it will return
         an array representing the function fit to data.

    Returns
    -------
    ndata: numpy.ndarray
         If normalize is True, it will return data normalized by the
         function fit to it.  If normalize is False, it will return
         an array representing the function fit to data.

    """

    if isinstance(mask, np.ndarray):
        if mask.shape != data.shape:
            raise ValueError('mask is not the same shape as data')
    else:
        raise TypeError('mask is not None or an numpy.ndarray')

    ys, xs = data.shape
    if isinstance(fitter, mod.fitting._FitterMeta):
        g_fit = fitter()
    else:
        raise TypeError('fitter is not a valid astropy.modeling.fitting')

    if not hasattr(func_init, 'n_inputs'):
        raise TypeError('func_init is not a valid astropy.modeling.model')

    if func_init.n_inputs == 2:
        y, x = np.indices((ys, xs))
        f = g_fit(func_init, x[mask], y[mask], data[mask])
        ndata = f(x, y)
    elif func_init.n_inputs == 1:
        ndata = 0.0 * data
        xarr = np.arange(xs)
        yarr = np.arange(ys)
        for i in range(xs):
            f = g_fit(func_init, yarr[mask[:, i]], data[:, i][mask[:, i]])
            ndata[:, i] = f(yarr)

    if normalize:
        return data / ndata * ndata.mean()
    else:
        return ndata


def fit_wavelength_solution(sol_dict):
    """Determine the best fit solution and re-fit each line with that solution

    The following steps are used to determine the best wavelength solution:
    1. The coefficients of the solution to each row are fit by a line
    2. The coefficients for each row are then replaced by the best-fit values
    3. The wavelenght zeropoint is then re-calculated for each row

    Parameters:
    -----------
    sol_dict: dict
        A dictionary where the key is the y-position of each row and the value
        is a list that containts an array of x values of peaks, the 
        corresponding wavelength array of the peaks, and a 
        `~astropy.modeling.model` that transforms between the x positions
        and wavelengths

    Returns: dict
    ------- 
    sol_dict: dict
        An updating dictionary with the new wavelength solution for each row

    """
    #determinethe quality of each solution
    weights = np.zeros(len(sol_dict))
    yarr = np.zeros(len(sol_dict))
    ncoef = len(sol_dict[sol_dict.keys()[0]][2].parameters)
    coef_list = []
    for i in range(ncoef):
        coef_list.append(np.zeros(len(sol_dict)))

    #populate the coeffient list with values
    for i, y in enumerate(sol_dict):
        yarr[i] = y
        mx, mw, ws = sol_dict[y]
        weights[i] = stats.median_absolute_deviation(ws(mx)-mw) / 0.6745
        for j, p  in enumerate(ws.parameters):
            coef_list[j][i] = p


    #fit each coefficient with a value
    coef_sol = []
    for coef in coef_list:
        fit_c = mod.fitting.LinearLSQFitter()
        c_init = mod.models.Polynomial1D(1)
        mask = (weights < 5*np.median(weights))
        c = iterfit1D(yarr[mask], coef[mask], fit_c, c_init, niter=7)
        coef_sol.append(c)

    #refit with only allowing zeropoint to change
    for i, y in enumerate(sol_dict):
        mx, mw, ws = sol_dict[y]
        for j, n  in enumerate(ws.param_names):
            c = coef_sol[j]
            ws.parameters[j] = c(y)

        weights = calc_weights(mx, mw, ws)
        dw = np.average(mw - ws(mx), weights=weights)
        ws.c0 = ws.c0 + dw
        sol_dict[y] = [mx, mw, ws]

    return sol_dict


def iterfit1D(x, y, fitter, model, yerr=None, thresh=5, niter=5):
    """Iteratively fit a function.  

    Outlyiers will have a reduced weight in the fit, and then 
    the fit will be repeated niter times to determine the
    best fits

    Parameters
    ----------
    x: numpy.ndarray
        Arrray of x-values

    y: numpy.ndarray
        Arrray of y-values

    fitter: ~astropy.modeling.fitting
        Method to fit the model

    model: ~astropy.modeling.model
        A model to be fit

    Returns
    -------
    m: ~astropy.modeling.model
        Model fit after reducing the weight of outlyiers

    """
    if yerr is None: yerr = np.ones_like(y)
    weights = np.ones_like(x)

    for i in range(niter):
        m = fitter(model, x, y, weights=weights)
        weights = calc_weights(x, y, m, yerr)

    return m


def calc_weights(x, y, m, yerr=None):
    """Calculate weights for each value based on deviation from best fit model

    Parameters
    ----------
    x: numpy.ndarray
        Arrray of x-values

    y: numpy.ndarray
        Arrray of y-values

    model: ~astropy.modeling.model
        A model to be fit
 
    yerr: numpy.ndarray
        [Optional] Array of uncertainties for the y-value

    Returns
    -------
    weights: numpy.ndarray
        Weights for each parameter

    """
    if yerr is None: yerr = np.ones_like(y)
    r = (y - m(x))/yerr
    s = np.median(abs(r - np.median(r))) / 0.6745
    biweight = lambda x: ((1.0 - x ** 2) ** 2.0) ** 0.5
    weights = 1.0/biweight(r / s)
    return weights



