HRSPROCESS
===========

hrsprocess includes steps for the basic CCD processing necessary for HRS data. 
The data can be reduced using the following syntax:

  >> from ccdproc import CCDData
  >> from pyhrs import hrs_process 
  >> hdu = CCDData.read('H201411170015.fits', units=u.adu) 
  >> hdu = hrs_process(hdu, overscan=[:,1:25], masterbias=masterbias, 
  ...                  gain_correct=True, flip=True )
  
This will return an image that has had the overscan corrected, trimmed, and positioned 
such that the orders increase from the bottom to the top and the dispersion goes from 
the left to the right. Flatfielding and calibration from a spectrophotometric standard 
will only be applied in later steps.

Processing Calibration Frames
-----------------------------
