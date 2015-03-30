
=====
PyHRS
=====

The PyHRS package is for the reduction of data from the High Resolution Spectrograph 
on the Southern African Large Telescope.   The goals of the package are
to provide tools to be able to produce scientific quality reductions for the 
the low, medium, and high resolution modes for HRS and to prepare data
for more specialized code for the reduction of high stability observations.

The package includes the following classes and functions:
- HRSModel
- hrsprocess
- HRSOrder
- hrstools


HRSModel
--------

HRSModel is a class for producing synthetic HRS spectra.  HRSModel is based 
on the `PySpectrograph.Spectrograph` class.  It only includes
a simple model based on the instrument confirguration and the spectrograph
equation.

hrs_process
-----------

`hrsprocess` includes steps for the basic CCD processing necessary for
HRS data.   It also includes steps necessary for creating calibration
frames.

HRSOrder
------------

HRSOrder is a class descirbe a single order from an HRS image.  The order then
has different tools for identifying regions, extracting orders, and defining 
properties of different orders such as wavelengths and calibrations.

hrsextract
----------

hrsextract includes all steps necessary to extract a single, one-dimensional
HRS spectrum. 

HRStools
--------

HRStools includes generally utilies used across different functions and classes.

Reference/API
=============

.. automodapi:: pyhrs



