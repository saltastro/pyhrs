import os
import sys
import numpy as np
import pickle

from scipy import signal 

from ccdproc import CCDData

import specutils
from astropy import units as u

from astropy import modeling as mod
from astropy.io import fits
from astropy import stats

import pylab as pl

import specreduce

from specreduce.interidentify import InterIdentify
from specreduce import spectools as st
from specreduce import WavelengthSolution
from specreduce import detect_lines
from specreduce import match_probability, ws_match_lines


from pyhrs import red_process, mode_setup_information
from pyhrs import zeropoint_shift, create_linelists, collapse_array
from pyhrs import HRSOrder, HRSModel


def get_spectra(arc, order_frame, n_order, soldir, target=True, flux_limit=100):
    shift_dict, ws = pickle.load(open(soldir+'sol_%i.pkl' % n_order))
    hrs = HRSOrder(n_order)
    hrs.set_order_from_array(order_frame.data)
    hrs.set_flux_from_array(arc.data, flux_unit=arc.unit)
    hrs.set_target(target)

    data, coef = hrs.create_box(hrs.flux, interp=True)

    pickle.dump(data, open('box_%s.pkl' % n_order, 'w'))
    xarr = np.arange(len(data[0]))
    warr = ws(xarr)
    flux = np.zeros_like(xarr)
    flux, shift_dict = collapse_array(data, i_reference=10)
    
    #smooth flux
    i_p = mod.models.Polynomial1D(3)
    fitter = mod.fitting.LinearLSQFitter()
    or_fitter = mod.fitting.FittingWithOutlierRemoval(fitter, stats.sigma_clip,
                                           niter=3, sigma=3.0)
    _t, p = or_fitter(i_p, xarr, flux)

    flux = flux-p(xarr)
    flux[flux<flux_limit] = flux_limit
    
    return xarr, warr, flux, ws, shift_dict

if __name__=='__main__':

   
    arc = CCDData.read(sys.argv[1])
    order_frame = CCDData.read(sys.argv[2], unit=u.adu)
    soldir = sys.argv[3]

    camera_name = arc.header['DETNAM'].lower()
    arm, xpos, target, res, w_c, y1, y2 = mode_setup_information(arc.header)

    n_min = order_frame.data[order_frame.data>0].min()
    for n_order in np.arange(n_min, order_frame.data.max()):
        print n_order
        if not os.path.isfile(soldir+'sol_%i.pkl' % n_order): continue
        x, w, f, ws, sh = get_spectra(arc, order_frame, int(n_order), soldir)
        m_arr = ws_match_lines(x, f, ws, dw=1.0, kernal_size=3)
        m, prob = match_probability(m_arr[:,1], m_arr[:,2], 
                            m_init=mod.models.Polynomial1D(1), 
                            fitter=mod.fitting.LinearLSQFitter(),  
                            tol=0.02, n_iter=5)
        ws = WavelengthSolution.WavelengthSolution(m_arr[:,0][prob>0.1], 
                                           m_arr[:,2][prob>0.1], 
                                           ws.model)
        ws.fit()
        pickle.dump([sh, ws], open('sol_%i.pkl' % n_order, 'w'))



