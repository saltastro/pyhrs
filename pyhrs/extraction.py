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

__all__=['extract_order', 'simple_extract_order', 'extract_science', 'normalize_order', 'normalize_spectra', 'stitch_spectra']


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
    for i in shift_dict.keys():
        if i < len(data) and i >= y1 and i <= y2:
            m = shift_dict[i]
            shift_flux = np.interp(xarr, m(xarr), data[i])
            if error is not None:
                shift_error = np.interp(xarr, m(xarr), error[i])
                # just in case flux is zero
                s = 1.0 * shift_flux
                s[s==0] = 0.0001
                w = (shift_error/s)**2
            else:
                shift_error = 1
                w = 1

            data[i] = shift_flux
            flux += shift_flux / w**2
            err += shift_error**2 / w**2
            weight += 1.0 / w**2
    #pickle.dump(data, open('box_%i.pkl' % n_order, 'w'))
    return warr, flux / weight, (err / weight)**0.5





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
    w,f,e = spectra_dict[o]
    xarr = np.arange(len(w))
    farr = np.zeros(len(w))
    for o in range(n_orders.min()+1, n_orders.max()+1):
        f = spectra_dict[o][1]
        f[np.isnan(f)] = 0
        farr += f
    f = fitter(model, xarr, farr)
    for o in range(n_orders.min()+1, n_orders.max()+1):
        spectra_dict[o][1] = spectra_dict[o][1] / f(xarr)  * f(xarr).mean()/spectra_dict[o][1].mean()
        spectra_dict[o][2] = spectra_dict[o][2] / f(xarr)  * f(xarr).mean()/spectra_dict[o][1].mean()
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
          w,f,e = spectra_dict[o]
          if np.all(np.isnan(f)): continue
          f[np.isnan(f)] = 0
          if normalize:
              f = normalize_order(w, f, model=model, fitter=fitter)
          if warr is None:
             warr = 1.0 * w
             farr = 1.0 * f
             earr = 1.0 * e
          else:
             warr = np.concatenate([warr, w])
             farr = np.concatenate([farr, f])
             earr = np.concatenate([farr, e])

    i = warr.argsort()
    return warr[i], farr[i], earr[i]
