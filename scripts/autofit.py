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

from specreduce import WavelengthSolution

from pyhrs import create_linelists
from pyhrs import collapse_array, match_lines
from pyhrs import HRSOrder, HRSModel




def autofit(arc, order_frame, n_order, ws, target='upper', interp=True, npoints=20, xlimit=1, slimit=0.5, wlimit=0.25):
    """Automatically identify lines in an arc and determine
       the fit 
    """
    sw, sf, slines, sfluxes = create_linelists('thar_list.txt', 'thar.fits')

    hrs = HRSOrder(n_order)
    hrs.set_order_from_array(order_frame.data)
    hrs.set_flux_from_array(arc.data, flux_unit=arc.unit)
    if target=='upper':
        hrs.set_target(True)
    else:
        hrs.set_target(False)
    data, coef = hrs.create_box(hrs.flux, interp=interp)

    #create a summed spectra by cross correlating each row
    xarr = np.arange(len(data[0]))
    flux, shift_dict = collapse_array(data, i_reference=10)
    mask = (sw > ws(xarr).min()-5) * ( sw <  ws(xarr).max()+5 )

    smask = (slines > ws(xarr).min()-10) * (slines < ws(xarr).max()+10)
    mx, mw = match_lines(xarr, flux, slines, sfluxes, ws, rw=3, npoints = npoints,
                         xlimit=xlimit, slimit=slimit, wlimit=wlimit)

    pl.plot(mx, mw, ls='', marker='o')
    pl.figure()
    fit_ws = mod.fitting.LinearLSQFitter()
    print n_order, ws.parameters,
    nws = fit_ws(ws, mx, mw)
    print nws.parameters
    ws = WavelengthSolution.WavelengthSolution(mx, mw, nws)
    ws.fit()

    pickle.dump([shift_dict, ws], open('tran_%i.pkl' % n_order, 'w'))
    #print ws.parameters
    #pl.plot(mx, mw-ws(mx), ls='', marker='o')
    #pl.figure()
    #pl.plot(ws(xarr), flux)
    #pl.plot(sw[mask], sf[mask] * flux.max()/sf[mask].max())
    #pl.show()
    return 



if __name__=='__main__':

   
    arc = CCDData.read(sys.argv[1])
    order_frame = CCDData.read(sys.argv[2], unit=u.adu)
    coef = pickle.load(open('coef.pkl'))

 
    camera_name = arc.header['DETNAM'].lower()

    if camera_name=='hrdet':
        arm = 'R'
        if arc.header['OBSMODE']=='HIGH RESOLUTION':
            target = True
        elif arc.header['OBSMODE']=='MEDIUM RESOLUTION':
            target = True
        elif arc.header['OBSMODE']=='LOW RESOLUTION':
            target = False
    else:
        arm = 'H'
        if arc.header['OBSMODE']=='HIGH RESOLUTION':
            target = True
        elif arc.header['OBSMODE']=='MEDIUM RESOLUTION':
            target = True
        elif arc.header['OBSMODE']=='LOW RESOLUTION':
            target = False

    n_order=65

    for n_order in range(54,83):
        #dc_dict, ws = pickle.load(open('sol_%i.pkl' % n_order))
        c_list = [coef[0](n_order), coef[1](n_order), coef[2](n_order), coef[3](n_order)]
        #ws.model.parameters = c_list
        #dw = np.median(ws.wavelength - ws(ws.x))
        #ws.model.c0 -= dw
    
        nws = mod.models.Legendre1D(3)
        nws.domain = [xarr.min(), xarr.max()]
        nws.parameters = c_list
    

        autofit(arc, order_frame, n_order, nws, target='upper', interp=True)

    #k = iws.keys()[0]
    #pickle.dump([dc_dict, iws[k]], open('sol_%i.pkl' % n_order, 'w'))


