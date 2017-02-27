# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the handling of different calibration files
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from astropy.extern import six
import warnings

from astropy import units as u
from astropy import modeling as mod
from astropy import stats

from scipy import ndimage as nd

from .hrstools import *
from .hrsorder import HRSOrder


__all__=['extract_order', 'simple_extract_order', 'extract_science', 'normalize_order', 
         'normalize_spectra', 'stitch_spectra', 'polyfitr', 'extract_normalize']


def extract_order(ccd, order_frame, n_order, ws, shift_dict, y1=3, y2=10, order=None, target=True, interp=False):
    """Given a wavelength solution and offset, extract the order

    Parameters
    ----------
    ccd: ~ccdproc.CCDData
        Science frame to be flatfielded

    order_frame: ~ccdproc.CCDData
        Frame containting the positions of each of the orders

    n_order: int
        Order to be extracted 

    ws: WavelengthSolution
        WavelengthSolution object containing the solution for this arc

    shift_dict: dict
        Dictionary containing the per row corrections to the spectra

    y1: int
        Minimum row to extract spectra

    y2: int
        Maximum row to extract spectra
 
    target_fiber: boolean
        Set to True to extract the target fiber

    interp: boolean
        Interpolate flux while rectifying order

    Returns
    -------
    spectra_dict: list
        Dictionary of spectra for each order

    """
    hrs = HRSOrder(n_order)
    hrs.set_order_from_array(order_frame.data)
    if ccd.uncertainty is None:
        error = None
    else:
       error = ccd.uncertainty.array
    hrs.set_flux_from_array(ccd.data, flux_unit=ccd.unit, error=error, mask=ccd.mask)

    # set pixels with bad fluxes to high numbers
    if hrs.mask is not None and hrs.error is not None:
        hrs.flux[hrs.mask] = 0
        hrs.error[hrs.mask] = 1000*hrs.error.mean()

    # set the aperture to extract
    hrs.set_target(target)

    # create the boxes of fluxes
    data, coef = hrs.create_box(hrs.flux, interp=interp)
    if hrs.error is not None:
        error, coef = hrs.create_box(hrs.error, interp=interp)
    else:
        error = None

    # create teh wavelength array and either use the
    # 1d or the 2d solution
    xarr = np.arange(len(data[0]))
    if order is None:
       warr = ws(xarr)
    else:
       warr = ws(xarr, order*np.ones_like(xarr))
    flux = np.zeros_like(xarr, dtype=float)
    err =  np.zeros_like(xarr, dtype=float)
    weight = 0
    fsum = np.zeros_like(xarr, dtype=float)
    for i in shift_dict.keys():
        if i < len(data) and i >= y1 and i <= y2:
            m = shift_dict[i]
            shift_flux = np.interp(xarr, m(xarr), data[i])
            if error is not None:
                shift_error = np.interp(xarr, m(xarr), error[i])
                # just in case flux is zero
                s = 1.0 * shift_flux
                s[s==0] = 0.0001
                w = 1.0 / shift_error**2
            else:
                shift_error = 1
                w = np.ones_like(xarr, dtype=float)

            data[i] = shift_flux
            flux += shift_flux * w
            fsum += shift_flux
            weight += w 
    #pickle.dump(data, open('box_%i.pkl' % n_order, 'w'))
    return warr, flux / weight, 1.0 / weight**0.5, fsum





def simple_extract_order(hrs, y1, y2, binsum=1, median_filter_size=None, interp=False):
    """Simple_extract_order transforms the observed spectra into a square form, 
       extracts all of the pixels between y1 and y2, and then sums the results.

    Parameters
    ----------
    hrs: ~HRSOrder
        An HRSOrder with a calibrated wavelength property

    y1: int
        Lower limit to extract the hrs data.  It should be in terms of the
        rectangular represenation of the flattened order

    y2: int
        Upper limit to extract the hrs data.  It should be in terms of the
        rectangular represenation of the flattened order

    binsum: int
        Amount of increase the binning in wavelength space

    median_filter_size: None or int
        Size for median filter to be run on the data and remove the general
        shape of the spectrum


    Returns
    -------
    wavelength: ~numpy.ndarray
        1D representation of the wavelength

    flux: ~numpy.ndarray
        1D representation of the flux
    """
    
    #set up the initial wavelength and flux 
    data, coef = hrs.create_box(hrs.flux, interp=interp)
    wdata, coef = hrs.create_box(hrs.wavelength)

    w1 = hrs.wavelength[hrs.wavelength>0].min()
    w2 = hrs.wavelength.max()
    dw = binsum * (w2-w1)/(len(data[0]))
    warr = np.arange(w1, w2, dw)
    farr = np.zeros_like(warr)

    y,x = np.indices(data.shape)
    for i in range(len(warr)):
        mask = (wdata >= warr[i]-0.5*dw) * (wdata < warr[i]+0.5*dw)
        mask = mask * (y >= y1) * (y < y2)
        if np.any(data[mask].ravel()):
            farr[i] = stats.sigma_clipped_stats(1.0*data[mask].ravel())[0]

    if median_filter_size:
        sf = nd.filters.median_filter(farr, size=median_filter_size)
        farr = farr / sf 
 
    #step to clean up the spectrum
    mask = (warr > 0) * (farr > 0) 
    return warr[mask], farr[mask]


def extract_science(ccd, wave_frame, order_frame, target_fiber=None,  extract_func=None, **kwargs):
    """Extract the spectra for each order in the order frame.  It will use the 
    extraction function specified by extract_func which will expect an 
    `~pyhrs.HRSObject` to be passed to it along with any arguments. 

    Parameters
    ----------
    ccd: ~ccdproc.CCDData
        Science frame to be flatfielded

    wave_frame: ~ccdproc.CCDData
        Frame containting the wavelength for each pixel

    order_frame: ~ccdproc.CCDData
        Frame containting the positions of each of the orders
 
    target_fiber: 'upper', 'lower', or None
        Specify the fiber to be extracted

    extract_fuc: function
        Fucntion to use for extracting the spectra
 

    Returns
    -------
    spectra_dict: list
        Dictionary of spectra for each order
       
    """
    spectra_dict={}

    #get a list of orders
    o1 = order_frame.data[order_frame.data>0].min()
    o2 = order_frame.data.max()
    order_arr = np.arange(o1, o2, dtype=int)

    for n_order in order_arr:
        hrs = HRSOrder(n_order)
        hrs.set_order_from_array(order_frame.data)
        if target_fiber=='upper': hrs.set_target(True)
        if target_fiber=='lower': hrs.set_target(False)
        hrs.set_flux_from_array(ccd.data, flux_unit=ccd.unit)
        hrs.set_wavelength_from_array(wave_frame.data, wavelength_unit=wave_frame.unit)
  
        if np.any(hrs.wavelength>0):
            w,f = extract_func(hrs, **kwargs)
            spectra_dict[n_order] = [w,f]
       
    return spectra_dict

def polyfitr(x, y, order, clip, xlim=None, ylim=None, mask=None, debug=False):
    """ Fit a polynomial to data, rejecting outliers.

    Fits a polynomial f(x) to data, x,y.  Finds standard deviation of
    y - f(x) and removes points that differ from f(x) by more than
    clip*stddev, then refits.  This repeats until no points are
    removed.

    Inputs
    ------
    x,y:
        Data points to be fitted.  They must have the same length.
    order: int (2)
        Order of polynomial to be fitted.
    clip: float (6)
        After each iteration data further than this many standard
        deviations away from the fit will be discarded.
    xlim: tuple of maximum and minimum x values, optional
        Data outside these x limits will not be used in the fit.
    ylim: tuple of maximum and minimum y values, optional
        As for xlim, but for y data.
    mask: sequence of pairs, optional
        A list of minimum and maximum x values (e.g. [(3, 4), (8, 9)])
        giving regions to be excluded from the fit.
    debug: boolean, default False
        If True, plots the fit at each iteration in matplotlib.

    Returns
    -------
    coeff, x, y:
        x, y are the data points contributing to the final fit. coeff
        gives the coefficients of the final polynomial fit (use
        np.polyval(coeff,x)).

    Examples
    --------
    >>> x = np.linspace(0,4)
    >>> np.random.seed(13)
    >>> y = x**2 + np.random.randn(50)
    >>> coeff, x1, y1 = polyfitr(x, y)
    >>> np.allclose(coeff, [1.05228393, -0.31855442, 0.4957111])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, order=1, xlim=(0.5,3.5), ylim=(1,10))
    >>> np.allclose(coeff, [3.23959627, -1.81635911])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, mask=[(1, 2), (3, 3.5)])
    >>> np.allclose(coeff, [1.08044631, -0.37032771, 0.42847982])
    True
    """

    x = np.asanyarray(x)
    y = np.asanyarray(y)
    isort = x.argsort()
    x, y = x[isort], y[isort]

    keep = np.ones(len(x), bool)
    if xlim is not None:
        keep &= (xlim[0] < x) & (x < xlim[1])
    if ylim is not None:
        keep &= (ylim[0] < y) & (y < ylim[1])
    if mask is not None:
        badpts = np.zeros(len(x), bool)
        for x0,x1 in mask:
            badpts |=  (x0 < x) & (x < x1)
        keep &= ~badpts

    x,y = x[keep], y[keep]
    if debug:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y,'.')
        ax.set_autoscale_on(0)
        pl.show()

    coeff = np.polyfit(x, y, order)
    if debug:
        pts, = ax.plot(x, y, '.')
        poly, = ax.plot(x, np.polyval(coeff, x), lw=2)
        pl.show()
        raw_input('Enter to continue')
    norm = np.abs(y - np.polyval(coeff, x))
    stdev = np.std(norm)
    condition =  norm < clip * stdev
    y = y[condition]
    x = x[condition]
    while norm.max() > clip * stdev:
        if len(y) < order + 1:
            raise Exception('Too few points left to fit!')
        coeff = np.polyfit(x, y, order)
        if debug:
            pts.set_data(x, y)
            poly.set_data(x, np.polyval(coeff, x))
            pl.show()
            raw_input('Enter to continue')
        norm = np.abs(y - np.polyval(coeff, x))
        stdev = norm.std()
        condition =  norm < clip * stdev
        y = y[condition]
        x = x[condition]

    return coeff,x,y
    
def extract_normalize(spectrum_file, order_to_extract, polyorder=3, sigmaclip=3.5, makeplot=False):
	""" This will extract a particular order from the reduced PyHRS spectrum files.
	
	Given a reduced PyHRS spectrum files (once data reduction has been perfomed)
	and an order to extract, this routine will try and fit a polynomial to the continuum
	and subtract it from the extracted order.
	
	Inputs
	------
	spectrum_file: string
		The reduced PyHRS spectrum filename.
	order_to_extract: int(3)
		The order number to extract.
	polyorder: int(2)
		The degree of the polynomial when fitting for the continuum.
	sigmaclip: float
		The sigma level to use when rejecting outlier datapoints in the
		polynomial fit.
	makeplot: boolean, default is False
		if True, plots the extract non-continuum corrected flux and the 
		contiuum corrected flux on two seperate figures.
	
	Returns
	------
	pyhrs_wavelength, final_pyhrs_flux:
		The wavelength and continuum correct flux of the extracted order
		is returned.
	
	Examples
	------
	>>>spectrum_file = pR201604180019_spec.fits
	>>>order_to_extract = 71
	>>>pyhrs_wavelength, pyhrs_norm_flux = extract_normalize(spectrum_file, order_to_extract, polyorder=3, sigmaclip=3.0, makeplot=False)
	
	"""
	
	sigma_clip = sigmaclip
	polyfit_order = polyorder
	img = spectrum_file
	hdu = fits.open(img)
	wave = hdu[1].data['Wavelength']
	flux = hdu[1].data['Flux']
	order = hdu[1].data['Order']
	order_to_plot = order_to_extract
	mask = (order==order_to_plot)
	pyhrs_wavelength = wave[mask].copy()
	pyhrs_fluxlvl = flux[mask].copy()
	## Try to fit a cotinuum to the backgraound and subtract this
	## This makes is much easier to fit the Gaussians
	py_coeff, py_C_Wave, py_C_offsets = polyfitr(pyhrs_wavelength, pyhrs_fluxlvl, order=polyfit_order, clip=sigma_clip)
	py_p = np.poly1d(py_coeff)
	xs = np.arange(min(pyhrs_wavelength), max(pyhrs_wavelength), 0.1)
	ys = np.polyval(py_p, xs)
	if makeplot:
		fig1 = plt.figure()
		plt.plot(pyhrs_wavelength, pyhrs_fluxlvl, color='green', linewidth=1.0, label='PyHRS Flux')
		plt.plot(xs, ys, color='red', label="Best Fit with outlier rejection")
		plt.plot(py_C_Wave, py_C_offsets, marker='*', color='red', linestyle="none", label="Points used in fit")
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux')
		plt.legend(scatterpoints=1)
		plt.show()

	pyhrs_fluxlvl = pyhrs_fluxlvl-np.polyval(py_p, pyhrs_wavelength)
	# First do some normalization on the spectrum
	final_pyhrs_flux = pyhrs_fluxlvl
	#final_pyhrs_flux = normalize(final_pyhrs_flux.reshape(1,-1), norm='l2')[0]

	if makeplot:
		fig2 = plt.figure()
		plt.plot(pyhrs_wavelength, final_pyhrs_flux, color='blue', linewidth=1.0, label='Continuum Corrected PyHRS Flux')
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux [normalized]')
		plt.legend(scatterpoints=1)
		plt.show()
		
	return pyhrs_wavelength, final_pyhrs_flux

def normalize_order(wavelength, flux, model=mod.models.Chebyshev1D(2), fitter=mod.fitting.LinearLSQFitter()):
    """Givne a spectra, fit a model to the spectra and remove it

    Parameters
    ----------
    wavelength: numpy.ndarray
       List of wavelenghts

    flux: numpy.ndarray
       flux of the array

    model: astropy.modeling.models
       Model for the continuum

    fitter: astropy.modeling.fitting
       Fitter used to fit the model

    Returns
    -------
    Normalized flux: numpy.ndarray
       Normalized flux for the spectra

    """
    f = fitter(model, wavelength, flux) 
    return flux/f(wavelength) 

def normalize_spectra(spectra_dict, model=mod.models.Chebyshev1D(2), 
                    fitter=None):
    """Givne a spectra, fit a model to the spectra and remove it

    Parameters
    ----------
    wavelength: numpy.ndarray
       List of wavelenghts

    flux: numpy.ndarray
       flux of the array

    model: astropy.modeling.models
       Model for the continuum

    fitter: astropy.modeling.fitting
       Fitter used to fit the model

    """
    n_orders = np.array(spectra_dict.keys(), dtype=int)
    o = n_orders.min()+1
    w,f,e,s = spectra_dict[o]
    xarr = np.arange(len(w))
    farr = np.zeros(len(w))
    for o in range(n_orders.min()+1, n_orders.max()+1):
        f = spectra_dict[o][1]
        f[np.isnan(f)] = 0
        farr += f
    f = fitter(model, xarr, farr)
    for o in range(n_orders.min()+1, n_orders.max()+1):
        spectra_dict[o][2] = spectra_dict[o][2] / f(xarr)  * f(xarr).mean()/spectra_dict[o][1].mean()
        spectra_dict[o][1] = spectra_dict[o][1] / f(xarr)  * f(xarr).mean()/spectra_dict[o][1].mean()
    #now do the summed spectra
    farr = np.zeros(len(w))
    for o in range(n_orders.min()+1, n_orders.max()+1):
        f = spectra_dict[o][3]
        f[np.isnan(f)] = 0
        farr += f
    f = fitter(model, xarr, farr)
    for o in range(n_orders.min()+1, n_orders.max()+1):
        spectra_dict[o][3] = spectra_dict[o][3] / f(xarr)  * f(xarr).mean()/spectra_dict[o][3].mean()

    return spectra_dict

def stitch_spectra(spectra_dict, n_min, n_max, normalize=False, model=None, fitter=None):
    """Give a spectra, stitch the spectra together

    Parameters
    ----------
    spectra_dict: dict
        Dictionary containing wavelenghts and fluxes

    normalize_order: bool
        Normalize the individual orders
  
    """
    warr = None
    for o in range(n_min, n_max):
          w,f,e,s = spectra_dict[o]
          if np.all(np.isnan(f)): continue
          f[np.isnan(f)] = 0
          if normalize:
              f = normalize_order(w, f, model=model, fitter=fitter)
          if warr is None:
             warr = 1.0 * w
             farr = 1.0 * f
             earr = 1.0 * e
             sarr = 1.0 * s
          else:
             warr = np.concatenate([warr, w])
             farr = np.concatenate([farr, f])
             earr = np.concatenate([farr, e])
             sarr = np.concatenate([farr, s])

    i = warr.argsort()
    return warr[i], farr[i], earr[i], sarr[i]
