import os
import sys
import numpy as np
import pickle

from ccdproc import CCDData

import specutils
from astropy import units as u

from astropy import modeling as mod
from astropy.io import fits

import pylab as pl

import specreduce

from specreduce.interidentify import InterIdentify
from specreduce import spectools as st
from specreduce import WavelengthSolution


from pyhrs import mode_setup_information
from pyhrs import zeropoint_shift
from pyhrs import HRSOrder, HRSModel

def write_spdict(outfile, sp_dict):
    
    o_arr = None
    w_arr = None
    f_arr = None

    for k in sp_dict.keys():
        w,f = sp_dict[k]
        if w_arr is None:
            w_arr = 1.0*w
            f_arr = 1.0*f
            o_arr = k*np.ones_like(w, dtype=int)
        else:
            w_arr = np.concatenate((w_arr, w))
            f_arr = np.concatenate((f_arr, f))
            o_arr = np.concatenate((o_arr, k*np.ones_like(w, dtype=int)))

    c1 = fits.Column(name='Wavelength', format='D', array=w_arr, unit='Angstroms')
    c2 = fits.Column(name='Flux', format='D', array=f_arr, unit='Counts')
    c3 = fits.Column(name='Order', format='I', array=o_arr)

    tbhdu = fits.BinTableHDU.from_columns([c1,c2,c3])
    tbhdu.writeto(outfile, clobber=True)

def extract_order(ccd, order_frame, n_order, ws, shift_dict, y1=3, y2=10, order=None, target=True, interp=False):
    """Given a wavelength solution and offset, extract the order

    """
    hrs = HRSOrder(n_order)
    hrs.set_order_from_array(order_frame.data)
    hrs.set_flux_from_array(ccd.data, flux_unit=ccd.unit)
    hrs.set_target(target)
    data, coef = hrs.create_box(hrs.flux, interp=interp)

    xarr = np.arange(len(data[0]))
    if order is None:
       warr = ws(xarr)
    else:
       warr = ws(xarr, order*np.ones_like(xarr))
    flux = np.zeros_like(xarr, dtype=float)
    weight = 0
    for i in shift_dict.keys():
        if i < len(data) and i >= y1 and i <= y2:
            m = shift_dict[i]
	    shift_flux = np.interp(xarr, m(xarr), data[i])
            data[i] = shift_flux
            flux += shift_flux * np.median(shift_flux)
            weight += np.median(shift_flux)
    pickle.dump(data, open('box_%i.pkl' % n_order, 'w'))
    return warr, flux/weight


def extract(ccd, order_frame, soldir, target='upper', interp=False):
    """Extract all of the orders and create a spectra table

    """
    if target=='upper': 
       target=True
    else:
       target=False

    if os.path.isdir(soldir):
       sdir=True
    else:
       sdir=False

    #set up the orders
    min_order = int(order_frame.data[order_frame.data>0].min())
    max_order = int(order_frame.data[order_frame.data>0].max())
    sp_dict = {}
    for n_order in np.arange(min_order, max_order):
        if sdir is True:
            if not os.path.isfile(soldir+'sol_%i.pkl' % n_order): continue 
            shift_dict, ws = pickle.load(open(soldir+'sol_%i.pkl' % n_order))
            w, f = extract_order(ccd, order_frame, n_order, ws, shift_dict, target=target, interp=interp)
        else:
            shift_all, ws = pickle.load(open(soldir))
            if n_order not in shift_all.keys(): continue
            w, f = extract_order(ccd, order_frame, n_order, ws, shift_all[n_order], order=n_order, target=target, interp=interp)

	sp_dict[n_order] = [w,f]
    return sp_dict


if __name__=='__main__':

   
    ccd = CCDData.read(sys.argv[1])
    order_frame = CCDData.read(sys.argv[2], unit=u.adu)
    soldir = sys.argv[3]

    rm, xpos, target, res, w_c, y1, y2 =  mode_setup_information(ccd.header)
    sp_dict = extract(ccd, order_frame, soldir, interp=True, target=target)
    outfile = sys.argv[1].replace('.fits', '_spec.fits')

    write_spdict(outfile, sp_dict)

