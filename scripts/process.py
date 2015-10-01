import sys, os
import numpy as np

from ccdproc import CCDData
from pyhrs.hrsprocess import *

from astropy import units as u

if __name__=='__main__':

   infile = sys.argv[1]
   mccd = sys.argv[2]
   if mccd=='None': 
      mccd=None
   else:
      mccd = CCDData.read(mccd, units=u.adu)
   if os.path.basename(infile).startswith('H'):
        ccd = blue_process(infile, masterbias=mccd)
   elif os.path.basename(infile).startswith('R'):
        ccd = red_process(infile, masterbias=mccd)
   else:
        exit('Are you sure this is an HRS file?')

   if len(sys.argv) == 5:
      order_frame = CCDData.read(sys.argv[3], unit=u.adu)
      flat_frame = CCDData.read(sys.argv[4])
      ccd=flatfield_science(ccd, flat_frame, order_frame, median_filter_size=None, interp=True)
      

   outfile = 'p'+infile
   ccd.write(outfile, clobber=True)

     
