# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy import units as u


class HRSOrder(object):

    """A class describing a single order for a High Resolutoin Spectrograph
    observation.

    Parameters
    -----------
    order: integer
        Order of the HRS observations

    region: list, tuple, or `~numpy.ndarray`
        region is an object that contains coordinates for pixels in
        the image which are part of this order.  It should be a list
        containing two arrays with the coordinates listed in each array.

    flux: `~numpy.ndarray`
        Fluxes corresponding to each pixel coordinate in region.

    wavelength: `~numpy.ndarray`
        Wavelengths corresponding to each pixel coordinate in region.

    order_type: str
        Type of order for the Order of the HRS observations

    flux_unit: `~astropy.units.UnitBase` instance or str, optional
        The units of the flux.

    wavelength_unit: `~astropy.units.UnitBase` instance or str, optional
        The units of the wavelength


    """

    def __init__(self, order, region=None, flux=None, wavelength=None,
                 flux_unit=None, wavelength_unit=None, order_type=None):
        self.order = order
        self.region = region
        self.flux = flux
        self.wavelength = wavelength

        self.flux_unit = flux_unit
        self.wavelength_unit = wavelength_unit
        self.order_type = order_type

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if not isinstance(value, int):
            raise TypeError('order is not an integer')
        self._order = value

    @property
    def order_type(self):
        return self._order_type

    @order_type.setter
    def order_type(self, value):
        if value not in ['sky', 'object', None]:
            raise TypeError("order_type is not None, 'sky', or 'object'")
        self._order_type = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        if value is None:
            self._region = None
            return

        if len(value) != 2:
            raise TypeError("region is not of length 2")
        if len(value[0]) != len(value[1]):
            raise TypeError(
                "coordinate lists in region are not of equal length")

        self.npixels = len(value[0])
        self._region = value

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, value):
        if value is None:
            self._flux = None
            return

        if self.region is None:
            raise ValueError('No region is set yet')

        if len(value) != self.npixels:
            raise TypeError("flux is not the same length as region")

        self._flux = value

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if value is None:
            self._wavelength = None
            return

        if self.region is None:
            raise ValueError('No region is set yet')

        if len(value) != self.npixels:
            raise TypeError("wavelength is not the same length as region")

        self._wavelength = value

    @property
    def flux_unit(self):
        return self._flux_unit

    @flux_unit.setter
    def flux_unit(self, value):
        if value is None:
            self._flux_unit = None
        else:
            self._flux_unit = u.Unit(value)

    @property
    def wavelength_unit(self):
        return self._wavelength_unit

    @wavelength_unit.setter
    def wavelength_unit(self, value):
        if value is None:
            self._wavelength_unit = None
        else:
            self._wavelength_unit = u.Unit(value)

    def set_order_from_array(self, data):
        """Given an array of data which has an order specified at each pixel,
           set the region at the given order for HRSOrder

        Parameters
        ----------
        data: `~numpy.ndarray`
            data is an 2D array with an order value specified at each pixel. If
            no order is available for a given pixel, the pixel should have a
            value of zero.

        """
        if not isinstance(data, np.ndarray):
            raise TypeError('data is not an numpy.ndarray')
        if data.ndim != 2:
            raise TypeError('data is not a 2D numpy.ndarray')

        self.region = np.where(data == self.order)

    def set_flux_from_array(self, data, flux_unit=None):
        """Given an array of data of fluxes, set the fluxes for
           the region at the given order for HRSOrder

        Parameters
        ----------
        data: `~numpy.ndarray`
            data is an 2D array with a flux value specified at each pixel.

        flux_unit: `~astropy.units.UnitBase` instance or str, optional
            The units of the flux.

        """

        if not isinstance(data, np.ndarray):
            raise TypeError('data is not an numpy.ndarray')

        if data.ndim != 2:
            raise TypeError('data is not a 2D numpy.ndarray')

        self.flux = data[self.region]
        self.flux_unit = flux_unit

    def set_wavelength_from_array(self, data, wavelength_unit):
        """Given an array of wavelengths, set the wavelength for
           each pixel coordinate in `~HRSOrder.region`.

        Parameters
        ----------
        data: `~numpy.ndarray`
            data is an 2D array with a wavelength value specified at each pixel

        wavelength_unit: `~astropy.units.UnitBase` instance or str, optional
            The units of the wavelength

        """

        if not isinstance(data, np.ndarray):
            raise TypeError('data is not an numpy.ndarray')

        if data.ndim != 2:
            raise TypeError('data is not a 2D numpy.ndarray')

        self.wavelength = data[self.region]
        self.wavelength_unit = wavelength_unit

    def set_wavelength_from_model(
            self, model, params, wavelength_unit, **kwargs):
        """Given an array of wavelengths, set the wavelength for
           each pixel coordinate in `~HRSOrder.region`.

        Parameters
        ----------
        model: function
            model is a callable function that will create a corresponding
            wavelength for each pixel in `~HRSOrder.region`.  The function
            can either be 1D or 2D.  If it is 2D, the x-coordinate should
            be the first argument.

        params: `~numpy.ndarray`
            Either a 1D or 2D list of parameters with the number of elements
            corresponding to the number of pixles. Typically, if model
            is a 1D function, this would be the x-coordinated from
            `~HRSOrder.region`.  Otherwise, this would be expected to be
            `~HRSOrder.region`.

        wavelength_unit: `~astropy.units.UnitBase` instance or str, optional
            The units of the wavelength

        **kwargs:
            All additional keywords to be passed to model

        """
        if not hasattr(model, '__call__'):
            raise TypeError('model is not a function')

        self.wavelength_unit = wavelength_unit

        if len(params) == self.npixels:
            self.wavelength = model(params, **kwargs)
        elif len(params) == 2:
            self.wavelength = model(params[1], params[0], **kwargs)

        else:
            raise TypeError('params is not the correct size or shape')


    def create_box(self, flux, interp=False):
        """Convert order into a square representation with integer shifts
           beteween each pixel

        Parameters
        ----------
        flux: ~numpy.ndarray
            Array of values to convert into a rectangular representation

        Returns
        -------
        box: ~numpy.ndarray
            Rectangular represnation of flux
        """
        xmax = self.region[1].max()
        xmin = 0
        ymax = self.region[0].max()
        ymin = self.region[0].min()
        xs = xmax-xmin
        coef = np.polyfit(self.region[1], self.region[0], 3)
        xarr = np.arange(xs+1)
        yarr = np.polyval(coef, xarr)-ymin
        x = self.region[1]-xmin
        y = self.region[0]-ymin - (np.polyval(coef, x) - ymin - yarr.min()).astype(int)
        ys = y.max()
        data = np.zeros((ys+1,xs+1))
        if interp:
           yarr = np.arange(ys+1)
           for i in range(xs):
               mask = (self.region[1]==i)
               ym = ymin-np.polyval(coef, xarr).min()
               if mask.sum() > 0:
                   y = self.region[0][mask]- np.polyval(coef, i) -  ym
                   data[:,i] = np.interp(yarr, y, flux[mask])
        else:
           data[y,x] = flux
        return data, coef
        

    def unravel_box(self, box):
        """Convert a rectangular represenation of the spectra back to a single
           array

        Parameters
        ----------
        box: ~numpy.ndarray
            Rectangular represnation of flux

        Returns
        -------
        data: ~numpy.ndarray
            Array of values to convert into a rectangular representation
        """
        xmax = self.region[1].max()
        xmin = 0
        ymax = self.region[0].max()
        ymin = self.region[0].min()
        ys = ymax-ymin
        xs = xmax-xmin
        data = np.zeros((ys+1,xs+1))
        coef = np.polyfit(self.region[1], self.region[0], 3)
        xarr = np.arange(xs+1)
        yarr = np.polyval(coef, xarr)-ymin
        x = self.region[1]-xmin
        y = self.region[0]-ymin - (np.polyval(coef, x) - ymin - yarr.min()).astype(int)
        data = np.zeros(self.npixels)
        data = box[y,x]
        return data
        

    def set_target(self, top_fiber=True):
        """Remove any pixels not associated with the target fiber. 

        Parameters
        ----------
        top_fiber: boolean
           If True, then the top fiber is the target fiber.  Otherwise,
           the target fiber is the one on bottom.
        """
        coef = np.polyfit(self.region[1], self.region[0], 3)
        if top_fiber:
           mask = (self.region[0] > np.polyval(coef, self.region[1]))  
        else:
           mask = (self.region[0] < np.polyval(coef, self.region[1]))  
       
        self.region = (self.region[0][mask], self.region[1][mask])

        if self.wavelength is not None:
            self.wavelength = self.wavelength[mask]
        if self.flux is not None:
            self.flux = self.flux[mask]

        return


    def extract_spectrum(self):
        """Extract 1D spectrum from the information provided so far and
           createa  `~specutils.Spectrum1D` object

        """
        try:
            from specutils import Spectrum1D
        except:
            from .spectrum1d import Spectrum1D

        if self.wavelength is None:
            raise ValueError('wavelength is None')
        if self.wavelength_unit is None:
            raise ValueError('wavelength_unit is None')
        if self.flux is None:
            raise ValueError('flux is None')
        if self.flux_unit is None:
            raise ValueError('flux_unit is None')

        wave = self.wavelength * self.wavelength_unit
        flux = self.flux * self.flux_unit
        return Spectrum1D.from_array(wave, flux)
