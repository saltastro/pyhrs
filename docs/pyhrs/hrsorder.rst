HRSORDER
===========

`HRSOrder` is a class describing a single echelle order.   

An `HRSorder` object can be initiated by 

    >>> from pyhrs.hrsorder import HRSOrder
    >>> h = HRSOrder(order = 63)

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
    >>> order_image = fits.open('HRDET_HR_Order.fits')
    >>> h.region = np.where(order_image.data==h.order)

Likewise, if the order is defined some other manner, the region can be defined
in a similar way using the same syntax.


Extracting the flux
-------------------

Once the region, the flux for the order can be easily set from the image array.  Simple
pass the data to `HRSOrder.set_flux_from_array` and the flux will be set for each of the
pixels in the region.

     >>> from astropy import units as u
     >>> data_image = fits.open('R201411140020.fits')
     >>> h.set_flux_from_array(data_image.data, flux_unit=u.electron)
     
The flux will now be accessible and only flux from pixels from region will be extracted. 
One the wavelength is set, a 1-D array can be extracted from the data using:
     
     >>> spectrum = h.extract_spectrum()

This will return a `~Spectrum1D` object that will be a one-dimenionsal
representation of the order with `wavelength` and `flux` propertiers.

Setting the Wavelength
----------------------

The wavelength can be set in two different ways.  If an array exists with wavelength specified
as a function of position, then the wavelength can be extracted in the same was as the flux:
 
    >>> wave _image = fits.open('HRDET_HR_Wavelength.fits')
    >>> h.set_wavelength_from_array(wave_image.data, wavelength_unit=u.Angstrom)

The other way that the wavelength can be set is via a model.  The model can either be a 1-D or
2-D model, but it should be a callable function of either x or x and y that returns a wavelength
value.  

    >>> from hrsmodel import HRSModel
    >>> hrs = HRSModel(order=63, camera_name='hrdet')
    >>> h.set_wavelength_from_model(hrs.get_wavelength, h.region[1], wavelength_unit=u.Angstrom)

For either case, each pixel coordinate in `region` will have a coresponding wavelength.

