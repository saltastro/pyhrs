import os, sys
import specutils
import numpy as np

import pylab as pl

# Read in and then print out a MIDAS spectra file.
# Only works with python 2.7 and specutils <= 0.2

s = specutils.io.read_fits.read_fits_spectrum1d(sys.argv[1])
flux = s.data
xarr = np.arange(len(flux))
warr = s._wcs(xarr)
for i in range(len(flux)):
    print warr[i], flux[i]

