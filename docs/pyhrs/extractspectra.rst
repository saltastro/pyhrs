Extracting Spectra
==================

The spectra is extracted from the image using the following steps:

1.  Each order is identified in the image and processed individually.  The order is corrected for curvature.   

2. The object and sky fibers are extracted separately.   

3. The flux from row is co-added after weighted by the error in each pixel.

4. The co-added flux is then normalized by the total weighting.  

This processes is repeated for each order to create the final spectra.  When written out to a FITS file, 
it includes a column for wavelength, flux, error, and order.  These steps are carried out by `~pyhrs.extract_order` 
and written to an FITS table by `~pyhrs.write_spdict`.
 
Normalization of the Spectra
============================

Prior to normalization, the sky fiber is subtracted from the object fiber.  Next, all of the fibers are averaged together and a polynomial is fit to the shape.  Finally, that polynomial is divided through each order. 

Stitching of the Orders
=======================

Currently the normalized orders are combined together into a single spectra.  No further processing
is done to stitch the spectra together. 





