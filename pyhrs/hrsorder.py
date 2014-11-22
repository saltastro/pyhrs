# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

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

       
    Raises
    ------
    TypeError
        If the `order` is not an integer


    """

    def __init__(self, order, region=None, flux=None, wavelength=None, order_type=None):
        self.order = order
        self.order_type = order_type

        if region is not None:
            self.region = region

        if flux is not None:
            self.flux = flux

        if wavelength is not None:
            self.wavelength = wavelength

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


