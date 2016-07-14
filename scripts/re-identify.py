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


from pyhrs import red_process, mode_setup_information
from pyhrs import zeropoint_shift, create_linelists, collapse_array
from pyhrs import HRSOrder, HRSModel


def identify(arc, order_frame, n_order, camera_name, xpos, ws=None, 
             target='upper', interp=True, w_c=None, 
             rstep=1, nrows=1, mdiff=20, wdiff=3, thresh=3, niter=5, dc=3, 
             ndstep=50, dsigma=5, method='Zeropoint', res=0.5, dres=0.01, 
             filename=None, smooth=0, inter=True, subback=False, 
             textcolor='green', log = None):
    """Run identify on a given order
    """
    sw, sf, slines, sfluxes = create_linelists('thar_list.txt', 'thar.fits')

    #this runs through the process to check the initial fit
    hrs = HRSOrder(n_order)
    hrs.set_order_from_array(order_frame.data)
    hrs.set_flux_from_array(arc.data, flux_unit=arc.unit)
    if target=='upper':
        hrs.set_target(True)
    else:
        hrs.set_target(False)
    data, coef = hrs.create_box(hrs.flux, interp=interp)

    pickle.dump(data, open('box_%s.pkl' % n_order, 'w'))
    

    #create a summed spectra by cross correlating each row
    xarr = np.arange(len(data[0]))
    flux = np.zeros_like(xarr)
    flux, shift_dict = collapse_array(data, i_reference=10)

    fdata = 0.0 * data
    fdata[10,:] = flux

    #set up the model
    fit_ws = mod.fitting.LinearLSQFitter()

    #set up the model for the spectrograph
    if ws is None:
        hrs_model = HRSModel(camera_name=camera_name, order=n_order)
        hrs_model.detector.xpos = xpos
        warr = hrs_model.get_wavelength(xarr) * u.mm
        warr = warr.to(u.angstrom).value
        warr = warr + w_c(xarr)
        ws_init = mod.models.Legendre1D(3)
        ws_init.domain = [xarr.min(), xarr.max()]
        nws = fit_ws(ws_init, xarr, warr)
        ws = WavelengthSolution.WavelengthSolution(xarr, warr, nws)
        ws.fit()
    else:
        warr = ws(xarr)
        

    istart = 10 
    smask = (slines > warr.min()-10) * (slines < warr.max()+10)
    function = 'poly'
    order = 3

    iws = InterIdentify(xarr, fdata, slines[smask], sfluxes[smask], ws, mdiff=mdiff, rstep=rstep,
              function=function, order=order, sigma=thresh, niter=niter, wdiff=wdiff,
              res=res, dres=dres, dc=dc, ndstep=ndstep, istart=istart,
              method=method, smooth=smooth, filename=filename,
              subback=subback, textcolor=textcolor, log=log, verbose=True)

    return shift_dict, iws


if __name__=='__main__':

   
    arc = CCDData.read(sys.argv[1])
    order_frame = CCDData.read(sys.argv[2], unit=u.adu)
    n_order = int(sys.argv[3]) #default order to use for initial file
    soldir = sys.argv[4]

    shift_dict, ws = pickle.load(open(soldir+'sol_%i.pkl' % n_order))

    camera_name = arc.header['DETNAM'].lower()

    arm, xpos, target, res, w_c, y1, y2 = mode_setup_information(arc.header)
    print arm, xpos, target
    dc_dict, iws = identify(arc, order_frame, n_order, camera_name, xpos, ws=ws,
             target=target, interp=True, w_c=w_c,
             rstep=1, nrows=2, mdiff=20, wdiff=3, thresh=3, niter=5, dc=3,
             ndstep=50, dsigma=5, method='Zeropoint', res=res, dres=res/10.0,
             filename=None, smooth=3, inter=True, subback=0,
             textcolor='black', log = None)

    k = iws.keys()[0]
    pickle.dump([dc_dict, iws[k]], open('sol_%i.pkl' % n_order, 'w'))


