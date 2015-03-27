Processing HRS Data
===================

`~pyhrs.hrsprocess` includes steps for the basic CCD processing necessary for HRS data. 
The code provides a wrapper for tasks from `ccdproc` to provide specific reductions
for HRS data.   In addition, it provides several functions for creating calibration
frames for the reduction of HRS data.

.. note::
    `hrsprcess` expects files to follow the SALT naming conventions


Processing Data frames
----------------------

Data frames can be process using the tasks `~pyhrs.blue_process` and `~pyhrs.red_process`.   The user
can select from several options included in these programs, but certain aspects are hard
wired to provide convenient functions for data reductions.  For example, to process blue
data: 

  >>> from pyhrs.hrsprocess import blue_process 
  >>> d('H201411170015.fits', units=u.adu) 
  >>> ccd = blue_process(hdu, masterbias=masterbias)
  
  
This will return an `ccdproc.CCDData` object that has had the overscan corrected, 
trimmed, gain corrected, had the master bias subtracted, and positioned 
such that the orders increase from the bottom to the top and the dispersion goes from 
the left to the right. Flatfielding and calibration from a spectrophotometric standard 
will only be applied in later steps.

Convenience functions for Processing Science Data:

* `~pyhrs.blue_process`: process data from the HRS blue camera
* `~pyhrs.red_process`: process data from the HRS red camera
* `~pyhrs.hrs_process`: convenience function for processing HRS data

The functions all pass appropriate parameters to `~~pyhrs.ccd_process`.  This tasks
wraps functions from `~ccdproc` for processing CCD images.   `~~pyhrs.ccd_process` has a number 
of steps, which are all optional, that include overscan subtraction, trimming, creating
error frames, masking, gain correction, and subtracting a master bias.  

Processing Calibration Frames
-----------------------------

Calibration frames can also be created using several convenience functions.  For example, 
passing a list of filenames to `create_masterbias` will process the data and combine them
to create the master bias frame.
  
  >>> from pyhrs.hrsprocess import create_masterbias
  >>> masterbias = create_masterbias(['H201411170015.fits', 'H201411170016.fits']
  
This will process each frame and return a masterbias `ccdproc.CCDData` object.   In addition,
there is a task for producing masterflats. 



.. _GitHub repo: https://github.com/saltastro/pyhrs
