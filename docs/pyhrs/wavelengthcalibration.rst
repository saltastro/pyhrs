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

1. Curvature due to the optical distortion is removed from the spectra and
   a square representation of the 2D spectra is created.  Only integer 
   shifts are applied to the data
2. A model of the spectrograph is created based on the order, camera, and
   xpos offset that are supplied.  A small correction described by a quadratic
   equation is added to the transformation calculated from the model.
3. In each row of the data, peaks are extracted and matched with a
   line in the atlas of wavelengths that is provided.  Due to the accuracy
   of the initial guess, lines are matched to within an angstrom and any
   line that might be blended is rejected. 
4. Once the first set of peaks and lines are matched up, a new solution
   is calculated for the given row.   Then the processes of matching
   lines and determining a wavelength solution is repeated using this new
   solution.  The best result from each line is saved.
5. Using all of the matched lines from all lines, a 'best' solution is 
   determined.   Everything but the zeroth order parameter of the fit
   is fixed to a slowly varying value based on the overall solution to all
   lines.  See `~pyhrs.fit_solution` for more details.
6. Based on the best solution found, the process is repeated for each
   row but only determing the zeropoint. 
7. Based on the solution found, a wavelength is assigned to each pixel

All of these steps are carried out by `~pyhrs.wavelength_calibrate_order`.  In
the end, this task returns an `~pyhrs.HRSOrder` object with wavelengths correspond
to every pixel where a good solution was found.   In addition, it also returns
the x-position, wavelength, and solution for the initial row.  


Calibrating an arc
------------------

For full automated calibration of an arc, `~pyhrs.wavelength_calibrate_arc` can be
used.   In this task, it applies `~pyhrs.wavelength_calibrate_order` to each 
of the orders in the frame.  It uses `pyhrs.HRSModel` for the first guess but takes
the quadratic correction from the solution of the nearest order.  It starts
with the initial order and the first row is also set by the user.  

