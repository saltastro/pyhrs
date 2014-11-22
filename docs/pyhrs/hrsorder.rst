HRSORDER
===========

`HRSOrder` is a class describing a single echelle order.   

An `HRSorder` object can be initiated by 

    >>> from pyhrs.hrsorder import HRSOrder
    >>> ord51 = HRSOrder(order = 51)

Properties of HRSOrder
----------------------

The two main properties of HRSOrder are the `order` and the `region`.  The 
`region` is a list of pixels in the image that are included in the order.   
Based on the region, additional properties can be derived included the flux
 of each of the pixels and the wavelength for each of the pixels.

Defining a Region
-----------------

The `HRSOrder.region` is defined by the user and passed to `HRSOrder` object.
The region is defined such that it should be a list, tuple, or 
`~numpy.ndarray`.  It should have two elements corresponding to the two 
dimensions of the image and follow the numpy convention for ordering of the
axis.

The region can be set by directly:
    >>> h.region = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]

If an image that defines the order already exists, then `numpy.where` can be 
used to set the order region 
    >>> from astropy.io import fits
    >>> order_image = fits.open(order_frame)
    >>> h.region = np.where(order_image.data==h.order)

Likewise, if the order is defined some other manner, the region can be defined
in a similar way using the same syntax.


Extracting the flux
-------------------

Setting the Wavelength
----------------------
