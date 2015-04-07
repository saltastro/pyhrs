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

__all__=['simple_extract_order']

def simple_extract_order(hrs, y1, y2, binsum=1, median_filter_size=None):
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
    data, coef = hrs.create_box(hrs.flux)
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


