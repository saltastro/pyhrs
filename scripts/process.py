import sys, os
import numpy as np

from ccdproc import CCDData
from pyhrs.hrsprocess import *

from astropy import units as u

import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process SALT HRS observations')
    parser.add_argument('infile', help='SALT HRS image')
    parser.add_argument('mccd', help='Master bias file')
    parser.add_argument('--o', dest='order', help='Master order file')
    parser.add_argument('--f', dest='flat', help='Master flat file')
    parser.add_argument('-s', dest='oscan', default=False, action='store_true', help='Apply overscan correction')
    parser.add_argument('-n', dest='cray', default=True, action='store_false', help='Do not cosmic ray clean')
    args = parser.parse_args()

    infile = args.infile
    mccd = CCDData.read(args.mccd, unit='electron')

    if os.path.basename(infile).startswith('H'):
         ccd = blue_process(infile, masterbias=mccd, oscan_correct=args.oscan)
    elif os.path.basename(infile).startswith('R'):
         ccd = red_process(infile, masterbias=mccd, oscan_correct=args.oscan)
    else:
         exit('Are you sure this is an HRS file?')

    if args.cray:
       from astroscrappy import detect_cosmics
       crmask, cleanarr = detect_cosmics(ccd.data, inmask=None, sigclip=4.5, sigfrac=0.3,
                   objlim=5.0, gain=1.0, readnoise=6.5,
                   satlevel=65536.0, pssl=0.0, niter=4,
                   sepmed=True, cleantype='meanmask', fsmode='median',
                   psfmodel='gauss', psffwhm=2.5, psfsize=7,
                   psfk=None, psfbeta=4.765, verbose=False) 
       ccd.data = cleanarr
       if ccd.mask == None:
          ccd.mask = crmask
       else:
          ccd.mask = ccd.mask * crmask
    if args.flat:
       order_frame = CCDData.read(args.order, unit=u.adu)
       flat_frame = CCDData.read(args.flat)
       ccd=flatfield_science(ccd, flat_frame, order_frame, median_filter_size=None, interp=True)
      

    outfile = 'p'+infile
    ccd.write(outfile, clobber=True)

     
