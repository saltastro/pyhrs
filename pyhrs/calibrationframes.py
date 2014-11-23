# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base processing for HRS data
#from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import numpy as np
from astropy.extern import six

from astropy import units as u
from astropy import modeling as mod

from scipy import ndimage as nd

import ccdproc

from .hrsprocess import *
from .hrstools import *

__all__ = ['create_masterbias', 'create_masterflat', 'create_orderframe']


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



def create_orderframe(data, first_order, xc, detect_kernal, fibers=2, func_order=3):
    """Create an order frame from from an observation. 

       A two dimensional detect_kernal is correlated with the image.  The kernal
       steps through y-space until a match is made.  Once a best fit is found, 
       the order is extracted to include pixels that may not be part of the 
       initial detection kernal.  Once all pixels have been extracted, they 
       are set to zero in the original frame.  The detection kernal is updated
       by the new order detected    

       Once one order is detected, it is then used to update the detection
       kernal, detect_kernal

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
       
    fibers: int
        The number of fibers in the system.  The norder will only be incremented
        after the number of fibers has been found.

    func_order: int
        The order for the polynomial to fit to the shape of the fibers


    Returns
    -------
    order_frame: ~numpy.ndarray
        An image with each of the order identified by their number
        

    """
    smooth_length=15
    ystart = 30
    y_limit = 3500

    #set up the arrays needed
    sdata = 1.0*data
    ndata = data[:,xc]
    order_frame = 0.0 * data

    #set up additiona information that we need
    ys, xs = data.shape
    xc = int(0.5*xs)
    norder = first_order

    #convolve with the default kernal
    cdata = np.convolve(ndata,detect_kernal, mode='same' )
    cdata *= (ndata>0)
    cdata = nd.filters.maximum_filter(cdata, smooth_length)

    #import pylab as pl
    #pl.plot(cdata[0:400])
    #print np.where(cdata==0)
    #pl.show()

    i = ystart
    nlen = len(detect_kernal)
    max_value = sdata.max()
    import pylab as pl
    #pl.plot(cdata)
    #pl.show()
    print nlen
    cdata[:ystart] = -1
    while i < ys:
        #find the highest peak in the convolution area
        y1 = max(0, i)
        y2 = y1 + nlen
        print 
        print y1, y2
        try:
            y2 = np.where(cdata==0)[0][0]
            #y2 = max(y2, c2)
            print y2
        except Exception, e:
            print(e)
            pass
        y2 = min(ys-1, y2)
        print y1, y2
        try:
            yc = cdata[y1:y2].argmax()+y1
        except: 
            break
        print y1, y2, yc
        #this is to make sure the two fibers
        #are both contained in the same 
        #order       
        sy1 = max(0, yc - 0.5*smooth_length)
        sy2 = min(ys-1, yc + 2*smooth_length)
        print 'smooth', yc, sy1, sy2
        sdata[sy1:sy2,xc] = max_value

        obj, nobj = nd.label(sdata>0.5*max_value)
        nobj = obj[yc,xc]
        order_frame += norder * (obj == nobj)

        #remove the data for this peak and 
        #all data it is now safe to ignore
        cdata[0:y2] = -1
        #now remove everything up until the next
        #peak 
        try:
            n2 = np.where(cdata>0)[0][0]
            cdata[0:n2] = -1
        except:
            n2 = y2
        #change the smoothing length
        z = (order_frame==norder)[:,xc]
        smooth_length  =  0.2*z[z==1].sum()

        i = n2
        norder += 1
        print i, norder, smooth_length
        if i > y_limit: 
            break
            sdata[order_frame > 0] = 0
            ndata = sdata[:,xc]
            detect_kernal = (order_frame==norder-1)[:,xc]
            detect_kernal = detect_kernal[detect_kernal>0]
            #smooth_length = smooth_length * len(detect_kernal)/nlen
            print nlen, len(detect_kernal), smooth_length
            nlen = len(detect_kernal)
            
            if nlen > 0:
                cdata = np.convolve(ndata,detect_kernal, mode='same' )
                cdata *= (ndata>0)
                cdata = nd.filters.maximum_filter(cdata, smooth_length)
                yn = np.where(cdata>0)[0][0]
                cdata[:yn] = -1
            y_limit = ys

            pl.plot(cdata)
            pl.show()

    return order_frame

