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


from pyhrs import red_process
from pyhrs import zeropoint_shift
from pyhrs import HRSOrder, HRSModel


def create_linelists(linefile, spectrafile):
    """Create line lists reads in the line list file in two different formats

    Parameters
    ----------
    linefile: str
        Name of file with wavelengths of arc lines

    spectrafile: str
        FITS file of a spectra of an arc lamp

    Returns
    -------
    slines: ~numpy.ndarray
        Arrary of wavelengths of arc lines

    sfluxes: ~numpy.ndarray
        Array of fluxes at each wavelength

    sw: ~numpy.ndarray
        array of wavelengths for arc spectra

    sf: ~numpy.ndarray
        array of fluxes for arc spectra

    """
    thar_spec = specutils.io.read_fits.read_fits_spectrum1d(spectrafile, dispersion_unit=u.angstrom)
    sw = thar_spec.wavelength.value
    sf = thar_spec.flux.value


    #read in arc lines
    slines = np.loadtxt(linefile, usecols=(0,), unpack=True)
    sfluxes = 0.0*slines

    for i in range(len(slines)):
        j = abs(thar_spec.wavelength-slines[i]*u.angstrom).argmin()
        sfluxes[i] = thar_spec.flux[j]

    return sw, sf, slines, sfluxes


def identify(arc, order_frame, n_order, camera_name, xpos, ws=None, 
             target='upper', interp=True, w_c=None, 
             rstep=1, nrows=1, mdiff=20, wdiff=3, thresh=3, niter=5, dc=3, 
             ndstep=50, dsigma=5, method='Zeropoint', res=0.5, dres=0.01, 
             filename=None, smooth=0, inter=True, subback=0, 
             textcolor='green', log = None):
    """Run identify on a given order
    """
    sw, sf, slines, sfluxes = create_linelists('thar_list.txt', 'thar.fits')

#master_redbias=CCDData.read('RBIAS.fits')
#arc = red_process('R201404290001.fits', masterbias=master_redbias)
#arc.write('pR201404290001.fits')
#exit()

#this runs through the process to check the initial fit

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
    flux = np.zeros_like(xarr)
    dc_dict={}
    for i in range(len(data)):
        dc, nnf = zeropoint_shift(xarr, data[i,:], xarr, data[10,:], dx=5.0, nx=100, center=4.0)
        dc_dict[i] = dc
        flux += nnf
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
        nws = fit_ws(ws_init, xarr, warr)
        ws = WavelengthSolution.WavelengthSolution(xarr, warr, nws)
        ws.fit()
        

    istart = 10 
    smask = (slines > warr.min()-10) * (slines < warr.max()+10)
    function = 'poly'
    order = 3

    iws = InterIdentify(xarr, fdata, slines[smask], sfluxes[smask], ws, mdiff=mdiff, rstep=rstep,
              function=function, order=order, sigma=thresh, niter=niter, wdiff=wdiff,
              res=res, dres=dres, dc=dc, ndstep=ndstep, istart=istart,
              method=method, smooth=smooth, filename=filename,
              subback=subback, textcolor=textcolor, log=log, verbose=True)

    return dc_dict, iws


if __name__=='__main__':

   
    arc = CCDData.read(sys.argv[1])
    order_frame = CCDData.read(sys.argv[2], unit=u.adu)
    n_order = int(sys.argv[3]) #default order to use for initial file

 
    camera_name = arc.header['DETNAM'].lower()

    res = 0.5
    w_c = None
    if camera_name=='hrdet':
        arm = 'R'
        if arc.header['OBSMODE']=='HIGH RESOLUTION':
            xpos = -0.025
            target = True
            res = 0.1
            w_c = mod.models.Polynomial1D(2, c0=0.440318305862, c1=0.000796335104265,c2=-6.59068602173e-07)
        elif arc.header['OBSMODE']=='MEDIUM RESOLUTION':
            xpos = 0.00
            target = True
            res = 0.2
        elif arc.header['OBSMODE']=='LOW RESOLUTION':
            xpos = -0.825
            target = False
            res = 0.4
            w_c = mod.models.Polynomial1D(2, c0=0.350898712753,c1=0.000948517538061,c2=-7.01229457881e-07)
    else:
        arm = 'H'
        if arc.header['OBSMODE']=='HIGH RESOLUTION':
            xpos = -0.025
            target = True
            res = 0.1
            w_c = mod.models.Polynomial1D(2, c0=0.840318305862, c1=0.000796335104265,c2=-6.59068602173e-07)
        elif arc.header['OBSMODE']=='MEDIUM RESOLUTION':
            xpos = 0.00
            target = True
            res = 0.2
        elif arc.header['OBSMODE']=='LOW RESOLUTION':
            xpos = 0.00
            target = False
            res = 0.4

    dc_dict, iws = identify(arc, order_frame, n_order, camera_name, xpos, ws=None,
             target=target, interp=True, w_c=w_c,
             rstep=1, nrows=2, mdiff=20, wdiff=3, thresh=3, niter=5, dc=3,
             ndstep=50, dsigma=5, method='Zeropoint', res=res, dres=res/10.0,
             filename=None, smooth=0, inter=True, subback=0,
             textcolor='green', log = None)

    k = iws.keys()[0]
    pickle.dump([dc_dict, iws[k]], open('sol_%i.pkl' % n_order, 'w'))


