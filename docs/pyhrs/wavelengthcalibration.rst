Wavelength Calibration
======================

Wavelength calibration requires the identification of known lines in a
spectrum of an arc to determine the transformation between pixel space
and wavelength.   

HRSModel
--------

`~pyhrs.HRSModel` is a class for producing synthetic HRS spectra.  HRSModel is based
on the `PySpectrograph.Spectrograph` class.  It includes
a simple model for both arms that is based on the instrument confirguration and the spectrograph
equation.  After adjusting for offsets in the fiber position relative to the CCD, the model
can return an approximation for the transformation that is accurate to within a few pixels. 

The residuals between the model and the actual solution though are well described by a
quadratic equation.   This quadratic does slowly vary accross the orders and is
different between the two arms.   Due to the change in the fiber position between
the different resolutions, this quadratic can change between the different configurations
as well.   

For these reasons, the initial guess for the wavelength solution is based on 
`~pyhrs.HRSModel` plus a quadratic correction.  The correction can either be
calculated manual or by automatically fitting a single row of an order. 

Calibrating a single order
--------------------------

To calibrate a single order, the following steps are carried out:

1. A single order is extracted. Curvature due to the optical distortion is 
   removed from the spectra and
   a square representation of the 2D spectra is created.  For best results,
   the data is interpolated to create the 2D representation.
2. Either the sky or the object fiber is extracted.   The extraction is
   carried out by co-adding the rows in the box corresponding to the 
   appropriate fiber. 
3. In each row of the data, peaks are extracted and matched with a
   line in the atlas of wavelengths that is provided.  Due to the accuracy
   of the initial guess, lines are matched to within an angstrom and any
   line that might be blended is rejected.
4. A model of the spectrograph is created based on the order, camera, and
   xpos offset that are supplied.  A small correction described by a quadratic
   equation is added to the transformation calculated from the model.  If available,
   a previous fit to the order is used instead of the model. 
5. In each row of the data, peaks are extracted and matched with a
   line in the atlas of wavelengths that is provided. This is either done
   manually or automatically using the [specreduce](https://github.com/crawfordsm/specreduce) 
   package.
6. Once lines have been identified in each of the orders, a 2D solution
   can be calcuated using all of the order and line information.


Calibrating an arc
------------------

For manual identification, the script `identify.py` can be used to identify
the lines in each arc.  If you want to repeat this, you can then use `re-identify.py`.
The script, `arc_solution.py` will produce automatic line identificaiton using
an already existing solution. 
