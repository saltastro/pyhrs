
=====
PyHRS
=====

The PyHRS package is for the reduction of data from the High Resolution Spectrograph 
on the Southern African Large Telescope.   The goals of the package are
to provide tools to be able to produce scientific quality reductions for the 
the low, medium, and high resolution modes for HRS and to prepare data
for more specialized code for the reduction of high stability observations.

The package includes the following classes and functions:
* HRSModel
* hrsprocess
* hrscalibrate
* hrsextract
* hrstools


HRSModel
--------

HRSModel is a class for producing synthetic HRS spectra.  HRSModel is based 
on the `PySpectrograph.Spectrograph` class.  It only includes
a simple model based on the instrument confirguration and the spectrograph
equation.

hrs_process
-----------

`hrsprocess` includes steps for the basic CCD processing necessary for
HRS data.   The data can be reduced using the following syntax:

    >> from astropy.io import fits
    >> from pyhrs import hrs_process
    >> hdu = fits.open('H201411170015.fits')
    >> hdu = hrs_process(hdu)

This will return an image that has had the overscan corrected, trimmed, and
positioned such that the orders increase from the bottom to the top and the
dispersion goes from the left to the right.  Flatfielding and calibration
from a spectrophotometric standard will only be applied in later steps.

hrscalibrate
------------

`hrscalibrate` includes steps for creating calibration frames necessary 
for HRS data.  This includes order maps, target/sky maps, and wavelength
calibration maps.

hrsextract
----------

hrsextract includes all steps necessary to extract a single, one-dimensional
HRS spectrum. 

HRStools
--------

HRStools includes generally utilies used across different functions and classes.



