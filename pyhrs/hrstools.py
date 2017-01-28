# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the tools used to process HRS data
#from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from scipy import signal
from scipy import ndimage as nd

from astropy import stats
from astropy import modeling as mod
from astropy import units as u
from astropy.io import fits


import specutils

__all__ = ['background', 'fit_order', 'normalize_image', 'xcross_fit', 'ncor',
           'iterfit1D', 'calc_weights', 'match_lines', 'zeropoint_shift', 
           'clean_flatimage', 'mode_setup_information', 'write_spdict',
           'fit_wavelength_solution', 'create_linelists', 'collapse_array']

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

def clean_flatimage(data, filter_size=101, flux_limit=0.1, block_size=100, percentile_low=30, median_size=5):
    """Remove flux from data that is not in an order

    This is an algorithm that removes inter-order flux from
    an image and any background.  The first step is to determine
    the position of the orders through a maximum filter of size
    filter size being passed over the data.  Any flux less than
    `flux_limit` * maximum flux will be removed.  The second
    step is to search in boxes of block_size and remove any flux
    which is lower than `percentile_low`  percentile in that box.  Finally
    the data is median filtered to further remove any structures.

    Parameters
    ----------
    data: numpy.ndarray
       2D array to be cleaned

    filter_size: int
       Size of filter to use for maximum_filter

    flux_limit: float
       Lower limit of maximum flux to retain

    block_size: int
       Size of box to search over data

    percentile_low: float
       Lower percentile to remove data within box

    median_size: int
       Size for median filter to pass over data
  
    Returns
    -------
    norm: numpy.ndarray
        cleaned image

    """
    norm = 1.0 * data
    for i in range(len(norm[0])):
        maxf = nd.filters.maximum_filter(norm[:,i], filter_size)
        mask = (norm[:,i] < flux_limit*maxf)
        norm[:,i][mask] = 0
  
    def _process_data(data, percentile_low=30, median_size=5):
        data = data - np.percentile(data, percentile_low)
        data[data<0] = 0
        return nd.filters.median_filter(data, median_size)

    i = 0
    while i < len(norm):
        j=0
        y1 = i
        y2 = min(i+block_size,  len(norm))
        while j < len(norm[0]):
            x1 = j
            x2 = min(j+block_size, len(norm[0]))
            norm[y1:y2, x1:x2] = _process_data(norm[y1:y2, x1:x2], percentile_low, median_size)
            j = j + block_size
        i = i + block_size


    return norm


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
        mask = (weights < 5 * np.median(weights))
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
    if s!=0: 
       weights = 1.0/biweight(r / s)
    else:
       weights = np.ones(len(x))
    return weights

def ncor(x, y):
    """Calculate the normalized correlation of two arrays

    Parameters
    ----------
    x: numpy.ndarray
        Arrray of x-values

    y: numpy.ndarray
        Arrray of y-values

    Returns
    -------
    ncor: float
        Normalize correctation value for two arrays

    """
    d=np.correlate(x,x)*np.correlate(y,y)
    if d<=0: return 0
    return np.correlate(x,y)/d**0.5


def xcross_fit(warr, farr, sw_arr, sf_arr, dw = 1.0, nw=100):
    """Calculate a zeropoint shift between the observed arc
       and the line list of values

    Parameters
    ----------
    warr: numpy.ndarray
        Estimated wavelength for arc

    farr: numpy.ndarray
        Flux at each arc position

    sw_arr: numpy.ndarray
        Wavelength of known lines 

    sf_arr: numpy.ndarray
        Flux of known lines

    dw: float 
        Value to search over. The search will be done from -dw to +dw

    nw: int
        Number of steps in the search

    Returns
    -------
    warr: numpy.ndarray
        Wavelength after correcting for shift from fiducial values
        
    
    """
    dw_arr = np.arange(-dw, dw, float(dw)/nw)
    cc_arr = 0.0 * dw_arr
    for i, w in enumerate(dw_arr):
        nsf_arr = np.interp(warr+w, sw_arr, sf_arr)
        cc_arr[i] = ncor(farr, nsf_arr)

    j = cc_arr.argmax()
    return warr+dw_arr[j]


def cross_match_arc(xarr, farr, sw, sf, ws, rw=10, dw=2.0, dres = 0.0001,
              npoints=20, xlimit=1.0, slimit=1.0):
    """Match lines in the spectra with specific wavleengths

    Match lines works by first identify the position of the brightest
    line in the image, fitting the line, and assigning wavelengths
    base on the match to the closest strong line.   

    Parameters
    ----------
    xarr: numpy.ndarray
       pixel positions

    farr: numpy.ndarray
       flux values at xarr positions

    sw: numpy.ndarray
       wavelengths of known arc lines

    sf: numpy.ndarray
       relative fluxes at those wavelengths

    ws: function 
       Function converting xarr into wavelengths.  It should be  
       defined such that wavelength = ws(xarr)

    rw: float
         Size of wavelength region to extract around peak

    dw: float
         Maximum wavelength shift to search over

    dres: float
         Sampling for creating the artificial spectra

    npoints: int
         The maximum number of points to bright points to fit.

    xlimit: float
         Maximum shift in line centroid when fitting

    slimit: float
         Minimum scale for line when fitting

    Returns
    -------
    warr: numpy.ndarray
        Wavelength values for each xarr position

    """
    fit_g = mod.fitting.LevMarLSQFitter()
    #flattenthe fluxes
    farr = farr - np.median(farr)

    # detect the lines in the image
    xp  = signal.find_peaks_cwt(farr, np.array([3]))
    if xp==[]: return None, None, None, None

    #set up the arrays
    xp = np.array(xp)
    fp = farr[xp]
    wp = ws(xp)
    warr = ws(xarr)
    mx = []
    mw = []

    sw_arr = np.arange(wp.min() - rw, wp.max() + rw, dres)
    sf_arr = 0.0 * sw_arr
    smask = (sw > wp.min() - rw) * (sw < wp.max() + rw)
    for w in sw[smask]:
        k = np.where(abs(sw_arr-w) < dres)
        sf_arr[k] += 1.0
    #smooth the artifical spectra
    sf_arr = np.convolve(sf_arr, np.ones(3/0.001), mode='same')

    all_mask = 0.0 * warr
    # find the brightest npoint lines in the image and match the wavelengths
    import datetime
    for i in fp.argsort()[::-1][:npoints]:
        mask = (warr > wp[i] - rw) * ( warr < wp[i]+rw)
        smask = (sw_arr > wp[i] - rw) * ( sw_arr < wp[i]+rw)

        warr[mask] = xcross_fit(warr[mask], farr[mask], sw_arr[smask],
                                sf_arr[smask], dw=dw, nw=10)

        #fit the best fitting line
        g = mod.models.Gaussian1D(amplitude=farr[mask].max(), mean=xp[i],
                                  stddev=0.5)
        g = fit_g(g, xarr[mask], farr[mask])
        x=g.mean.value
        w_x = np.interp(x, xarr[mask], warr[mask])
        s=3*g.stddev.value*ws.c1
        j = abs(sw - w_x).argmin()

        #reject things that are not good fits or are too narrow
        if abs(x-xp[i]) < xlimit and g.stddev.value > slimit:
           mx.append(x)
           mw.append(sw[j])
        else:
           pass

        all_mask[mask] += 1

    return warr, (all_mask>0), mx, mw

def match_lines(xarr, farr, sw, sf, ws, rw=5, npoints=20, xlimit=1.0, slimit=1.0,
                wlimit=1.0):
    """Match lines in the spectra with specific wavleengths

    Match lines works by finding the closest peak based on the x-position
    transformed by ws that is within wlimit of a known line.

    Parameters
    ----------
    xarr: numpy.ndarray
       pixel positions

    farr: numpy.ndarray
       flux values at xarr positions

    sw: numpy.ndarray
       wavelengths of known arc lines

    sf: numpy.ndarray
       relative fluxes at those wavelengths

    ws: function 
       Function converting xarr into wavelengths.  It should be  
       defined such that wavelength = ws(xarr)

    rw: float
         Radius in pixels around peak to extract for fitting the center

    npoints: int
         The maximum number of points to bright points to fit.

    xlimit: float
         Maximum shift in line centroid when fitting

    slimit: float
         Minimum scale for line when fitting

    wlimit: float
         Minimum separation in wavelength between peak and line

    Returns
    -------
    mx: numpy.ndarray
        x-position for matched lines

    mw: numpy.ndarray
        Wavelength position for matched lines

    """

    fit_g = mod.fitting.LevMarLSQFitter()
    #flattenthe fluxes
    farr = farr - np.median(farr)

    # detect the lines in the image
    xp  = signal.find_peaks_cwt(farr, np.array([3]))
    if xp==[]: return [], []

    #set up the arrays
    xp = np.array(xp)
    fp = farr[xp]
    wp = ws(xp)
    warr = ws(xarr)
    mx = []
    mw = []

    for i in fp.argsort()[::-1][0:npoints]:
        gmask = abs(xarr-xp[i]) < rw
        g = mod.models.Gaussian1D(amplitude=farr[gmask].max(), mean=xp[i],
                                  stddev=0.5)
        g = fit_g(g, xarr[gmask], farr[gmask])
        x=g.mean.value
        if abs(x-xp[i]) < xlimit and g.stddev.value > slimit:
            w = ws(x) 
            if wlimit is None:
                j = abs(sw-w).argmin()
                mx.append(x)
                mw.append(sw[j])
            else:
                mask = abs(sw-w) < wlimit
                l = sw[mask]
                if len(l)==1:
                   mx.append(x)
                   mw.append(sw[mask][0])
    return mx, mw

        
def zeropoint_shift(xarr, flux, reference_xarr, reference_flux, dx=5.0, nx=100, center=None):
    """Determine the shift between two spectra and return the re-interpolated
       spectra

    Parameters
    ----------
    xarr: numpy.ndarray
        x-positions for spectra

    flux: numpy.ndarray
        Flux values for spectra

    reference_xarr: numpy.ndarray
        x-positions for reference spectra

    reference_flux: numpy.ndarray
        Flux values for reference spectra

    dx: float
        range of x-values to search over
 
    nx: int
        number of steps for dx

    center: None or int
        If specified, it will interpolate over the dx values to calculate the
        best shift.  Otherwise, it will just use the dx value with the largest
        cross correlation value.

    Returns
    -------
    dc: float
        shift in x-position for the spectra

    shift_flux: numpy.ndarray
        Shifted spectra to frame of reference spectra

    """
    dc_arr = np.arange(-dx, dx, 1.0*dx/nx)
    nc_arr = np.zeros_like(dc_arr)
    for i, dc in enumerate(dc_arr):
        shift_flux = np.interp(reference_xarr+dc, xarr, flux)
        nc_arr[i] = ncor(reference_flux, shift_flux)
        
    if center:
        dc = dc_arr[nc_arr.argmax()]
        mask = abs(dc_arr-dc) < 1.0*center*dx/nx
        m_init = mod.models.Polynomial1D(2)
        m_fit = mod.fitting.LinearLSQFitter()
        m = m_fit(m_init, dc_arr[mask], nc_arr[mask])
        dc = m.parameters[1]/2/-m.parameters[2]
    else:
        dc = dc_arr[nc_arr.argmax()]
    shift_flux = np.interp(reference_xarr+dc, xarr, flux)
    return dc, shift_flux

def create_linelists(linefile, spectrafile):
    """Create line lists reads in the line list file in two different formats

    Parameters
    ----------
    linefile: str
        Name of file with wavelengths of arc lines

    spectrafile: str
        FITS file of a spectra of an arc lamp

    Returns
    -------
    slines: ~numpy.ndarray
        Arrary of wavelengths of arc lines

    sfluxes: ~numpy.ndarray
        Array of fluxes at each wavelength

    sw: ~numpy.ndarray
        array of wavelengths for arc spectra

    sf: ~numpy.ndarray
        array of fluxes for arc spectra

    """
    thar_spec = specutils.io.read_fits.read_fits_spectrum1d(spectrafile, dispersion_unit=u.angstrom)
    sw = thar_spec.wavelength.value
    sf = thar_spec.flux.value


    #read in arc lines
    slines = np.loadtxt(linefile, usecols=(0,), unpack=True)
    sfluxes = 0.0*slines

    for i in range(len(slines)):
        j = abs(thar_spec.wavelength-slines[i]*u.angstrom).argmin()
        sfluxes[i] = thar_spec.flux[j]

    return sw, sf, slines, sfluxes


def collapse_array(data, i_reference):
    """Given an array, determine the best shift between each row and then co-add

    Parameters
    ----------
    data: ~numpy.ndarray
        A 2D array for an image of a single fiber

    i_reference: int
        Row to which match the other rows


    Returns 
    -------
    flux: ~numpy.ndarray
        Co-addition of all rows  in data after finding the appropriate
        shift for each row.

    """
    xarr = np.arange(len(data[0]))
    flux = np.zeros_like(xarr, dtype=float)

    #set up the reference positions
    y0 = data[i_reference,:]
    xp = np.array(signal.find_peaks_cwt(y0, np.array([3])))
    fp = y0[xp]
    m_init = mod.models.Polynomial1D(2)
    fit_m = mod.fitting.LinearLSQFitter()
    m = fit_m(m_init, xp, xp)
    #this step is just run to really get the centroid positions
    xp, wp = match_lines(xarr, y0, xp, fp, m, npoints = 50, xlimit=5, slimit=0.1, wlimit=5)
    x0 = np.array(xp)

    #now find the match for each row 
    shift_dict={}
    for i in np.arange(len(data)):
        y =  data[i,:]
        xp, m0 = match_lines(xarr, y, x0, fp, m, npoints = 50, xlimit=5, slimit=0.1, wlimit=5)
        if len(xp) < 2: continue
        m = iterfit1D(xp, m0, fit_m, m_init)
        shift_dict[i] = m.copy()
        shift_flux = np.interp(xarr, m(xarr), y)
        flux += shift_flux
    return flux, shift_dict


def mode_setup_information(header): 
    """Return information needed for reductions for a given mode
       based on the header 
 
    Parameters
    ----------
    header: dict
       Header information for the image
 
    Returns
    -------
    arm: str
        prefix for image

    xpos: float

    target: string
        Whether the target is in the upper or lower fiber

    res: 
        An estimate for the resolution element for the mode

    w_c: ~astropy.modeling.models
        Correction to model for wavelength solution

    """
    if header['DETNAM'].lower()=='hrdet':
        arm = 'R'
        if header['OBSMODE']=='HIGH RESOLUTION':
            xpos = -0.025
            target = 'upper'
            res = 0.1
            w_c = mod.models.Polynomial1D(2, c0=0.440318305862, c1=0.000796335104265,c2=-6.59068602173e-07)
            y1 = 4 
            y2 = 28
        elif header['OBSMODE']=='MEDIUM RESOLUTION':
            xpos = 1.325
            target = 'upper'
            res = 0.2
            w_c = mod.models.Polynomial1D(2, c0=-0.566869781923, c1=0.00136529716199, 
                              c2=-6.36217218931e-07)
            y1 = 3 
            y2 = 25

        elif header['OBSMODE']=='LOW RESOLUTION':
            xpos = -0.825
            target = 'lower'
            res = 0.4
            w_c = mod.models.Polynomial1D(2, c0=0.350898712753,c1=0.000948517538061,c2=-7.01229457881e-07)
            y1 = 2 
            y2 = 14
    else:
        arm = 'H'
        if header['OBSMODE']=='HIGH RESOLUTION':
            xpos = -0.025
            target = 'upper'
            res = 0.1
            w_c = mod.models.Polynomial1D(2, c0=0.840318305862, c1=0.000796335104265,c2=-6.59068602173e-07)
            y1 = 3 
            y2 = 21
        elif header['OBSMODE']=='MEDIUM RESOLUTION':
            xpos = 1.55
            target = 'upper'
            res = 0.2
            w_c = mod.models.Polynomial1D(2, c0=-0.26996285172, c1=0.000936845602323, c2=-5.97067772021e-07)
            y1 = 3 
            y2 = 21

        elif header['OBSMODE']=='LOW RESOLUTION':
            xpos = -0.30
            target = 'lower'
            res = 0.4
            w_c = mod.models.Polynomial1D(2, c0=-0.0933573480342, c1=0.00101532206108, c2=-9.39770670751e-07)
            y1 = 2 
            y2 = 9

    return arm, xpos, target, res, w_c, y1, y2


def write_spdict(outfile, sp_dict, header=None):
    """Write out a spectral dictionary

    Parameters
    ----------
    outfile: str
       Name of outfile

    sp_dict: dict
       Dictionary containing wavelength, flux, and error as a function of order

    header: None or ~astropy.io.fits.header
       Optional header for outfile

    """

    o_arr = None
    w_arr = None
    f_arr = None
    e_arr = None
    s_arr = None

    for k in sp_dict.keys():
        w, f, e, s = sp_dict[k]
        if w_arr is None:
            w_arr = 1.0*w
            f_arr = 1.0*f
            e_arr = 1.0*e
            s_arr = 1.0*s
            o_arr = k*np.ones_like(w, dtype=int)
        else:
            w_arr = np.concatenate((w_arr, w))
            f_arr = np.concatenate((f_arr, f))
            e_arr = np.concatenate((e_arr, e))
            s_arr = np.concatenate((s_arr, s))
            o_arr = np.concatenate((o_arr, k*np.ones_like(w, dtype=int)))

    c1 = fits.Column(name='Wavelength', format='D', array=w_arr, unit='Angstroms')
    c2 = fits.Column(name='Flux', format='D', array=f_arr, unit='Counts')
    c3 = fits.Column(name='Order', format='I', array=o_arr)
    c4 = fits.Column(name='Error', format='D', array=e_arr, unit='Counts')
    c5 = fits.Column(name='Sum', format='D', array=s_arr, unit='Counts')

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5])
    prihdu = fits.PrimaryHDU(header=header)
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(outfile, clobber=True)

