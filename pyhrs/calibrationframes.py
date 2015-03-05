# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the handling of different calibration files
from __future__ import (absolute_import, division,# print_function,
                        unicode_literals)

import os
import numpy as np
from astropy.extern import six
import warnings

from astropy import units as u
from astropy import modeling as mod

from scipy import ndimage as nd


import ccdproc

from .hrsprocess import *
from .hrstools import *
from .hrsmodel import HRSModel

__all__ = ['create_masterbias', 'create_masterflat', 'create_orderframe',
           'wavelength_calibrate_arc', 'wavelength_calibrate_order']


def create_masterbias(image_list):
    """Create a master bias frame from a list of images

    Parameters
    ----------
    image_list: list
        List contain the file names to be processed

    Returns
    -------
    masterbias: ccddata.CCDData
        Combine master bias from the biases supplied in image_list

    """
    # determine whether they are red or blue
    if os.path.basename(image_list[0]).startswith('H'):
        func = blue_process
    elif os.path.basename(image_list[0]).startswith('R'):
        func = red_process
    else:
        raise TypeError('These are not standard HRS frames')

    # reduce the files
    ccd_list = []
    for image_name in image_list:
        ccd = func(image_name, masterbias=None, error=False)
        ccd_list.append(ccd)

    # combine the files
    cb = ccdproc.Combiner(ccd_list)
    nccd = cb.median_combine(median_func=np.median)

    return nccd


def create_masterflat(image_list, masterbias=None):
    """Create a master flat frame from a list of images

    Parameters
    ----------
    image_list: list
        List contain the file names to be processed

    masterbias: None, `~numpy.ndarray`,  or `~ccdproc.CCDData`
        A materbias frame to be subtracted from ccd.

    Returns
    -------
    masterflat: ccddata.CCDData
        Combine master flat from the flats supplied in image_list

    """
    # determine whether they are red or blue
    if os.path.basename(image_list[0]).startswith('H'):
        func = blue_process
    elif os.path.basename(image_list[0]).startswith('R'):
        func = red_process
    else:
        raise TypeError('These are not standard HRS frames')

    # reduce the files
    ccd_list = []
    for image_name in image_list:
        ccd = func(image_name, masterbias=masterbias, error=False)
        ccd_list.append(ccd)

    # combine the files
    cb = ccdproc.Combiner(ccd_list)
    nccd = cb.median_combine(median_func=np.median)

    return nccd


def create_orderframe(data, first_order, xc, detect_kernal, smooth_length=15,
                      y_start=0, y_limit=None):
    """Create an order frame from from an observation.

    A two dimensional detect_kernal is correlated with the image.  The
    kernal steps through y-space until a match is made.  Once a best fit is
    found, the order is extracted to include pixels that may not be part of
    the initial detection kernal.  Once all pixels have been extracted, they
    are set to zero in the original frame.  The detection kernal is updated
    by the new order detected

    Parameters
    ----------
    data: ~numpy.ndarray
        An image with the different orders illuminated.  Any processing of this
        image should have been performed prior to running `create_orderframe`.

    first_order: int
        The first order to appear in the image starting from the bottom of the
        image

    xc: int
        The x-position to extract a 1-D map of the orders

    detect_kern: ~numpy.ndarray
        The initial detection kernal which have the shape of a single order.

    smooth_length: int
        The length to smooth the images by prior to processing them

    y_start: int
        The initial value to start searching for the first maximum

    y_limit: int
        The limit in y-positions for automatically finding the orders.


    Returns
    -------
    order_frame: ~numpy.ndarray
        An image with each of the order identified by their number

    Notes
    -----
    Currently no orders are extrcted above y_limit and the code still needs to
    be updated to handle those higher orders

    """
    # set up the arrays needed
    sdata = 1.0 * data
    ndata = data[:, xc]
    order_frame = 0.0 * data

    # set up additiona information that we need
    ys, xs = data.shape
    xc = int(0.5 * xs)
    norder = first_order

    if y_limit is None:
        y_limit = ys

    # convolve with the default kernal
    cdata = np.convolve(ndata, detect_kernal, mode='same')
    cdata *= (ndata > 0)
    cdata = nd.filters.maximum_filter(cdata, smooth_length)

    i = y_start
    nlen = len(detect_kernal)
    max_value = sdata.max()
    cdata[:y_start] = -1
    while i < ys:
        # find the highest peak in the convolution area
        y1 = max(0, i)
        y2 = y1 + nlen
        try:
            y2 = np.where(cdata == 0)[0][0]
        except Exception as e:
            warnings.warn(str(e))
        y2 = min(ys - 1, y2)
        try:
            yc = cdata[y1:y2].argmax() + y1
        except:
            warning.warn('Breaking at (0)'.format(i))
            break
        # this is to make sure the two fibers
        # are both contained in the same
        # order
        sy1 = max(0, yc - 0.5 * smooth_length)
        sy2 = min(ys - 1, yc + 2 * smooth_length)
        sdata[sy1:sy2, xc] = max_value

        obj, nobj = nd.label(sdata > 0.5 * max_value)
        nobj = obj[yc, xc]
        order_frame += norder * (obj == nobj)

        # remove the data for this peak and
        # all data it is now safe to ignore
        cdata[0:y2] = -1
        # now remove everything up until the next
        # peak
        try:
            n2 = np.where(cdata > 0)[0][0]
            cdata[0:n2] = -1
        except:
            n2 = y2
        # change the smoothing length
        z = (order_frame == norder)[:, xc]
        smooth_length = 0.2 * z[z == 1].sum()

        i = n2
        norder += 1
        if i > y_limit:
            break
            # TODO: Something needs to be figured out for when above the limit
            sdata[order_frame > 0] = 0
            ndata = sdata[:, xc]
            detect_kernal = (order_frame == norder - 1)[:, xc]
            detect_kernal = detect_kernal[detect_kernal > 0]
            nlen = len(detect_kernal)

            if nlen > 0:
                cdata = np.convolve(ndata, detect_kernal, mode='same')
                cdata *= (ndata > 0)
                cdata = nd.filters.maximum_filter(cdata, smooth_length)
                yn = np.where(cdata > 0)[0][0]
                cdata[:yn] = -1
            y_limit = ys

    return order_frame

def wavelength_calibrate_arc():
    """Wavelength calibrate an arc spectrum from HRS
 
    """
    return


def wavelength_calibrate_order(hrs, slines, sfluxes, ws_init, fit_ws):
    """Wavelength calibration of a  single order from the HRS arc spectra

    The calibration proceeds through following steps:
    1. Curvature due to the optical distortion is removed from the spectra and
       a square representation of the 2D spectra is created.  Only integer 
       shifts are applied to the data
    2. A model of the spectrograph is created based on the order, camera, and
       xpos offset that are supplied.
    3. In each row of the data, peaks are extracted and matched with a
       line in the atlas of wavelengths that is provided (slines, sflux).  For
       the details of the matching process, see the match_arc function.
    4. Once the first set of peaks and lines are matched up, a new solution
       is calculated for the given row.   Then the processes of matching
       lines and determining a wavelength solution is repeated.  The best
       result from each line is saved.
    5. Using all of the matched lines from all lines, a 'best' solution is 
       determined.   Everything but the zeroth order parameter of the fit
       is fixed to a slowly varying value based on the overall solution to all
       lines.  See fit_solution for more details.
    6. Based on the best solution to each line, the wavelength is determined in
       each pixel of the data.   The wavelength property in the hrs object is 
       opdated and the new hrs object is returned. 

    Parameters
    ----------
    hrs: ~HRSOrder
        Object describing a single HRS order.  It should already contain the
        defined order and the flux from the arc for that order

    sw: numpy.ndarray
       wavelengths of known arc lines

    sf: numpy.ndarray
       relative fluxes at those wavelengths

    ws_init: ~astropy.modeling.model
        A initial model decribe the trasnformation from x-position to 
        wavelength

    fit_ws: ~astropy.modeling.fitting
        Method to fit the model

 
    Returns
    -------
    hrs: ~HRSOrder
        An HRSOrder with a calibrated wavelength property

    """
    import datetime
    import pickle

    #create the box
    xmax = hrs.region[1].max()
    xmin = 0 
    ymax = hrs.region[0].max()
    ymin = hrs.region[0].min()
    ys = ymax-ymin
    xs = xmax-xmin
    data = np.zeros((ys+1,xs+1))
    ydata = np.zeros((ys+1,xs+1))
    coef = np.polyfit(hrs.region[1], hrs.region[0], 3)
    xarr = np.arange(xs+1)
    yarr = np.polyval(coef, xarr)-ymin

    for i in range(hrs.npixels):
        y,x = hrs.region[0][i]-ymin, hrs.region[1][i]-xmin
        y = y - int(np.polyval(coef, x) - ymin - yarr.min())
        data[y,x] = hrs.flux[i]
        ydata[y,x] = hrs.region[0][i]
    pickle.dump(data, open('box_%i.pkl' % hrs.order, 'w'))

    #set the wavelength
    func_order = len(ws_init.parameters)
    warr = ws_init(xarr)
    
    
    #match the lines
    y = data[:,int(0.5*len(xarr))]
    y = np.where(y>0)[0]
    nmax = y.max()
    thresh=3

    #find the best solution
    y0 = 50
    farr = 1.0*data[y0,:]
    farr = farr[::-1]
    mx, mw = match_lines(xarr, farr, slines, sfluxes, ws_init, wlimit=0.5)
    ws = iterfit1D(mx, mw, fit_ws, ws_init)

    sol_dict={}
    for y in range(0, nmax, 1):
        farr = 1.0*data[y,:]
        farr = farr[::-1]
        if farr.sum() > 0:
            mx, mw = match_lines(xarr, farr, slines, sfluxes, ws, wlimit=0.5)
            if len(mx) > func_order:
                 nws = iterfit1D(mx, mw, fit_ws, ws_init, thresh=thresh)
                 sol_dict[y] = [mx, mw, nws]
    pickle.dump(sol_dict, open('sol_%i.pkl' % hrs.order, 'w'))
    sol_dict = fit_wavelength_solution(sol_dict)

    #update the wavelength values
    wdata = 0.0*data
    for y in sol_dict: 
        mx, mw, nws = sol_dict[y]
        wdata[y,:] = nws(xarr.max() - xarr)
 
    x = hrs.region[1]
    y = hrs.region[0] - ymin  - (np.polyval(coef, hrs.region[1]) - ymin - yarr.min()).astype(int)
    hrs.wavelength = wdata[y,x]
    

   
    return hrs
