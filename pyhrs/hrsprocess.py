# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base processing for HRS data
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.extern import six

from astropy import units as u

import ccdproc

__all__ = ['hrs_process', 'create_masterbias', 'create_masterflat']

def hrs_process(ccd, oscan=None, trim=None, error=False, masterbias=None,
                bad_pixel_mask=None, gain=None, rdnoise=None, oscan_median=True,
                oscan_model=None):
    """hrs_process performs the basic ccd processing on data from the High
       Resolution Spectrograph from the Southern African Large Telescope.  The
       user has the option of what steps to include but the steps that are 
       currently available include:
       * overscan correction
       * trimming of the image
       * subtraction of master bias
       * gain correction
       * image alignment

       The task returns a processed `ccdproc.CCDData` object. 

    Parameters
    ----------
    ccd: `ccdproc.CCDData`
        Frame to be reduced

    oscan: None, str, or, `~ccdproc.ccddata.CCDData`
        For no overscan correction, set to None.   Otherwise proivde a region
        of `ccd` from which the overscan is extracted, using the FITS
        conventions for index order and index start, or a 
        slice from `ccd` that contains the overscan. 

    trim: None or str
        For no trim correction, set to None.   Otherwise proivde a region
        of `ccd` from which the image should be trimmed, using the FITS
        conventions for index order and index start.

    error: boolean
        If True, create an uncertainty array for ccd

    masterbias: None, `~numpy.ndarray`,  or `~ccdproc.CCDData`
        A materbias frame to be subtracted from ccd.

    bad_pixel_mask: None or `~numpy.ndarray`
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels havea value of 1 and good pixels a value of 0.

    gain: None or `~astropy.Quantity`
        Gain value to multiple the image by to convert to electrons

    rdnoise: None or `~astropy.Quantity`
        Read noise for the observations.  The read noise should be in 
        `~astropy.units.electron`

    flip: boolean
        If True, the image will be flipped such that the orders run from the
        bottom of the image to the top and the dispersion runs from the left
        to the right.

    oscan_median :  bool, optional
        If true, takes the median of each line.  Otherwise, uses the mean

    oscan_model :  `~astropy.modeling.Model`, optional
        Model to fit to the data.  If None, returns the values calculated
        by the median or the mean.

    Returns
    -------
    ccd: `ccdproc.CCDData`
        Reduded ccd

    Examples
    --------

    1. To overscan, trim, and gain correct a data set:
 
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from hrsprocess import hrs_process
    >>> ccd = CCDData(np.ones([100, 100]), unit=u.adu)
    >>> nccd = hrs_process(ccd, oscan='[1:10,1:100]', trim='[10:100, 1,100]',
                           error=False, gain=2.0*u.electron/u.adu)


    """
    #make a copy of the object
    nccd = ccd.copy()
 
    #apply the overscan correction
    if isinstance(oscan, ccdproc.CCDData):
        nccd = ccdproc.subtract_overscan(nccd, overscan=oscan, median=oscan_median, model=oscan_model)
    elif isinstance(oscan, six.string_types):
        nccd = ccdproc.subtract_overscan(nccd, fits_section=oscan, median=oscan_median, model=oscan_model)
    elif oscan is None:
        pass
    else:
        raise TypeError('oscan is not None, a string, or CCDData object')

    #apply the trim correction
    if isinstance(trim, six.string_types):
        nccd = ccdproc.trim_image(nccd, fits_section=trim)
    elif trim is None:
        pass
    else:
        raise TypeError('trim is not None or a string')

    #create the error frame
    if error and gain is not None and rdnoise is not None:
       nccd = ccdproc.create_deviation(nccd, gain=gain, rdnoise=rdnoise)
    elif error and (gain is None or rdnoise is None):
       raise ValueError('gain and rdnoise must be specified to create error frame')
 
    #test subtracting the master bias
    if isinstance(masterbias, ccdproc.CCDData):
        nccd = nccd.subtract(masterbias)
    elif isinstance(masterbias, np.ndarray):
        nccd.data = nccd.data - masterbias
    elif masterbias is None:
        pass
    else:
        raise TypeError('masterbias is not None, numpy.ndarray,  or a CCDData object')
      
    #apply the bad pixel mask
    if isinstance(bad_pixel_mask, np.ndarray):
       nccd.mask = bad_pixel_mask
    elif bad_pixel_mask is None:
        pass
    else:
        raise TypeError('bad_pixel_mask is not None or numpy.ndarray')

    #apply the gain correction
    if isinstance(gain, u.quantity.Quantity):
        nccd = ccdproc.gain_correct(nccd, gain)
    elif gain is None:
        pass
    else:
        raise TypeError('gain is not None or astropy.Quantity')



    return nccd

def create_masterbias():
    return 

def create_masterflat():
    return
