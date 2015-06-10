
import sys
from astropy.io import fits
import glob
from pylab import *






figure()

img = sys.argv[1]
hdu = fits.open(img)
wave = hdu[1].data['Wavelength']
flux = hdu[1].data['Flux']
order = hdu[1].data['Order']
for o in sys.argv[2:]:
   o = int(o)
   mask = (order==o)
   plot(wave[mask], flux[mask])
show()
