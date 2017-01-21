import os
import sys
import argparse
import numpy as np
import pickle

from ccdproc import CCDData

import specutils
from astropy import units as u

from astropy import modeling as mod
from astropy.io import fits


import pylab as pl


from pyhrs import mode_setup_information
from pyhrs import zeropoint_shift
from pyhrs import HRSOrder, HRSModel
from pyhrs import extract_order

def write_spdict(outfile, sp_dict, header=None):
    
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
    prihdu = fits.PrimaryHDU(header=header)
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(outfile, clobber=True)


def extract(ccd, order_frame, soldir, target='upper', interp=False, twod=False):
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
        if sdir is True and twod is False:
            if not os.path.isfile(soldir+'sol_%i.pkl' % n_order): continue 
            shift_dict, ws = pickle.load(open(soldir+'sol_%i.pkl' % n_order))
            w, f = extract_order(ccd, order_frame, n_order, ws, shift_dict, target=target, interp=interp)

        if sdir is False and twod is False:
            sol_dict = pickle.load(open(soldir, 'rb'))
            if n_order not in sol_dict.keys(): continue
            ws, shift_dict = sol_dict[n_order]
            w, f = extract_order(ccd, order_frame, n_order, ws, shift_dict, target=target, interp=interp)

        if sdir is False and twod is True:
            shift_all, ws = pickle.load(open(soldir))
            if n_order not in shift_all.keys(): continue
            w, f = extract_order(ccd, order_frame, n_order, ws, shift_all[n_order], order=n_order, target=target, interp=interp)

	sp_dict[n_order] = [w,f]
    return sp_dict


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Excract SALT HRS observations')
    parser.add_argument('infile', help='SALT HRS image')
    parser.add_argument('order', help='Master order file')
    parser.add_argument('soldir', help='Master bias file')
    parser.add_argument('-2', dest='twod', default=False, action='store_true', help='2D solution')
    args = parser.parse_args()

    ccd = CCDData.read(args.infile)
    order_frame = CCDData.read(args.order, unit=u.adu)
    soldir = args.soldir

    rm, xpos, target, res, w_c, y1, y2 =  mode_setup_information(ccd.header)
    sp_dict = extract(ccd, order_frame, soldir, interp=True, target=target, twod=args.twod)
    outfile = sys.argv[1].replace('.fits', '_spec.fits')

    write_spdict(outfile, sp_dict, header=ccd.header)

