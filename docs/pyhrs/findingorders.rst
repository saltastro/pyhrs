Finding Orders
==============

A critical step to the reduction of HRS data is finding orders in the images.
Typically, images of flat fields or of a bright object can be used to identify
orders in the frame.   The data must first be processed.   Unfortunately, a
truly perfect data set for the identification of the orders is rarely 
available as the response function across the CCD can widely vary. 

Below we outline some tasks that can be used for identifying orders in HRS
images and the steps that are used to process the data to make the 
identification possible. 


Normalizing a flat field frame
------------------------------

Due to the changing response function across the CCD, a fiber flat image can show
significant vignetting in both the vertical and horizontal direction.   To remove 
the vignetting in the vertical direction, the `~pyhrs.normalize_image` task can be used.

The `~pyhrs.normalize_image` task fits a function to a fiber flat image after masking the
image.  Either a 1D or 2D function can be fit to the image, and the fitting function should
be specified by the user.   For the best performance, it is best to either apply an already
existing order frame or to smooth the image and mask areas of low response.

Here is an example of the steps needed to normalize an image:

    >>> from astropy.io import fits
    >>> from astropy import modeling as mod
    >>> from scipy import ndimage as nd
    >>> from pyhrs import normalize_image

    >>> hdu = fits.open('HFLAT.fits')
    >>> image = nd.filters.maximum_filter(image, 10)
    >>> mask = (image > 500)
    >>> norm = normalize_image(image, mod.models.Legendre1D(10), mask=mask)
    >>> norm[norm<10000] = 0

This will produce a `~numpy.ndarry` where good areas all have the same value and bad areas
will have values set to zero.   This significantly simplifies the process of identifying 
orders in the image.   However, orders at the very top of the image with little or no signal
will still be difficult to detect.

Creating an order frame 
-----------------------

The next step is to create an order frame.  An order frame is defined as an image where each pixel
associated with an order is identified and labeled with that order.   To produce the order frame,
an initial detection kernal (based on a user input) is convolved with a single column in the image.
The first maximum identified is assocatied with the initial input order given by the user.  To 
identify the full 2D shape of the order, all pixels above a certain threshold and connected are
identified.   These pixels are then given the value of the initial order.   Once the order
is identified, all pixels associated with this order are set to zero.  The detection 
kernal is then updated based on the 1D shape of this order, the order is incremented, and the 
process is repeated until all orders are identified in the frame.   

All of these steps are accomplished running the `~pyhrs.create_orderframe` task.  An example of running
this task is the follow:
   
    >>> norm[norm>500] = 500
    >>> xc = int(norm.shape[0]/2)
    >>> detect_kern = norm[30:110, xc]
    >>> frame = create_orderframe(norm, 84, xc, detect_kern, y_start=30, y_limit=4050)

This will produce the order frame for the blue arm in medium resolution mode.  It will identify all orders
up to the limit of y-position of 4050.   For the red arm, the first order is 53 for the medium resolution
mode.  

