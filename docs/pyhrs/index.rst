****************************
HRS Data Reduction (`pyhrs`)
****************************

Introduction
=============

.. note::
    `pyhrs` works only with astropy version 1.0.0 or later.  It also requires ccdproc version 0.3.0 or later.  


The `pyhrs` package provides steps for reducing and extracting data from the High Resolution Spectrograph on the Southern African Large Telescope.  The package includes the following sub-packages to help with the processing and reduction of the data:

+ A class describing a single HRS order, `~pyhrs.HRSOrder`, that includes position, wavelength, and flux properties
+ A class describing the HRS spectrograph, `~pyhrs.HRSModel`, to allow for accurate modeling of the HRS Spectrograph
+ Raw data can be processed using `~pyhrs.hrsprocess` for both the blue and red arms.   
+ Orders can be identified in the images using `~pyhrs.create_orderframe`
+ `~pyhrs.wavelength_calibrate_arc` can be used to calculate the wavelength calibration.

Once the data are reduced and calibrated, extraction of the object spectra can proceed based on the prefered method of the user.

Processing your data
====================

For more information about how to process your data, please check out

.. toctree::
    :maxdepth: 1
    
    hrsprocess
    findingorders

.. _GitHub repo: https://github.com/saltastro/pyhrs
    


