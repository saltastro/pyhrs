# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module tests the HRSModel class
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from astropy import units as u

from ..hrsorder import HRSOrder

def create_hrsorder():
    h = HRSOrder(order=16)
    y = np.arange(25)
    y = y.reshape(5,5)
    h.set_order_from_array(y)
    h.set_flux_from_array(y, flux_unit=u.electron)
    def f(x,y): return 2*x+y
    h.set_wavelength_from_model(f, h.region, wavelength_unit=u.nm)
    return h


def test_hrsorder_empty(): 
    with pytest.raises(TypeError):
        h = HRSOrder()

def test_hrsorder():
    h = HRSOrder(order=67)
    assert h.order == 67


#test setting it with an order
def test_hrsorder_bad():
    with pytest.raises(TypeError):
        h = HRSOrder(order=37.5)

#test order type
def test_hrsorder_order_type_object():
    h = HRSOrder(order=37, order_type='object')
    assert h.order_type == 'object'

def test_hrsorder_order_type_sky():
    h = HRSOrder(order=37, order_type='sky')
    assert h.order_type == 'sky'

def test_hrsorder_order_type_None():
    h = HRSOrder(order=37, order_type='sky')
    h.order_type = None
    assert h.order_type == None 

def test_hrsorder_order_type_bad():
    with pytest.raises(TypeError):
        h = HRSOrder(order=37, order_type='badtype')

#test defining a region
def test_hrsorder_region():
    r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
    h = HRSOrder(order=37, region = r)
    assert h.region == r

def test_hrsorder_region_length_bad():
    r = [(3,3,3,4,4,4,5,5,5)]
    with pytest.raises(TypeError):
        h = HRSOrder(order=37, region=r)

def test_hrsorder_region_pixels_bad():
    r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2)]
    with pytest.raises(TypeError):
        h = HRSOrder(order=37, region=r)

#test setting the flux
def test_hrsorder_flux():
    r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
    f = np.arange(len(r[0]))
    h = HRSOrder(order=37, region = r, flux = f)
    assert_array_equal(h.flux, f)

def test_hrsorder_flux_length():
    with pytest.raises(TypeError):
        r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
        f = np.arange(len(r[0]-1))
        h = HRSOrder(order=37, region = r, flux = f)

def test_hrsorder_flux_noregion():
    with pytest.raises(ValueError):
        r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
        f = np.arange(len(r[0]))
        h = HRSOrder(order=37, flux=f)

#test setting the flux unit
def test_hrsorder_flux_unit():
    h = HRSOrder(order=37, flux_unit=u.electron)
    assert h.flux_unit == u.electron

#test setting the flux unit
def test_hrsorder_wavelength_unit():
    h = HRSOrder(order=37, wavelength_unit=u.nm)
    assert h.wavelength_unit == u.nm


#test setting the wavelength
def test_hrsorder_wavelength():
    r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
    w = np.arange(len(r[0]))
    h = HRSOrder(order=37, region = r, wavelength = w)
    assert_array_equal(h.wavelength, w)

def test_hrsorder_wavelength_length():
    with pytest.raises(TypeError):
        r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
        w = np.arange(len(r[0]-1))
        h = HRSOrder(order=37, region = r, wavelength = w)

def test_hrsorder_wavelength_noregion():
    with pytest.raises(ValueError):
        r = [(3,3,3,4,4,4,5,5,5), (1,2,3,1,2,3,1,2,3)]
        w = np.arange(len(r[0]))
        h = HRSOrder(order=37, wavelength = w)



#test setting the array from the data
def test_hrsorder_set_order_from_array():
     h = HRSOrder(order=16) 
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     assert h.region == (np.array([3]), np.array([1]))
     assert y[h.region] == [16]

def test_hrsorder_set_order_from_array_baddata():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     with pytest.raises(TypeError):
        h.set_order_from_array(5)

def test_hrsorder_set_order_from_array_baddata_shape():
     h = HRSOrder(order=16)
     y = np.arange(25)
     with pytest.raises(TypeError):
        h.set_order_from_array(y)

#test setting the flux from the data
def test_hrsorder_set_flux_from_array():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     h.set_flux_from_array(y)
     assert h.flux == [16]

def test_hrsorder_set_flux_from_array_baddata():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     with pytest.raises(TypeError):
        h.set_flux_from_array(5)
     
def test_hrsorder_set_flux_from_array_baddata_shape():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     with pytest.raises(TypeError):
        h.set_flux_from_array(np.ones(10))

#test setting the wavelength from the data
def test_hrsorder_set_wavelength_from_array():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     h.set_wavelength_from_array(y, wavelength_unit=u.nm)
     assert h.wavelength == [16]

def test_hrsorder_set_wavelength_from_array_baddata():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     with pytest.raises(TypeError):
        h.set_wavelength_from_array(5, wavelength_unit=u.nm)

def test_hrsorder_set_wavelength_from_array_baddata_shape():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     with pytest.raises(TypeError):
        h.set_wavelength_from_array(np.ones(10), wavelength_unit=u.nm)

#test setting the wavelength from the data
def test_hrsorder_set_wavelength_from_model1D():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     def f(x): return 2*x
     h.set_wavelength_from_model(f, h.region[1], wavelength_unit=u.nm)
     assert_array_equal(h.wavelength, 2*h.region[1])

def test_hrsorder_set_wavelength_from_model2D():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     def f(x,y): return 2*x+y
     h.set_wavelength_from_model(f, h.region, wavelength_unit=u.nm)
     assert_array_equal(h.wavelength, 2*h.region[1]+h.region[0])


def test_hrsorder_set_wavelength_from_model_badparam():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     def f(x,y): return 2*x+y
     with pytest.raises(TypeError):
          h.set_wavelength_from_model(f, 5, wavelength_unit=u.nm)

def test_hrsorder_set_wavelength_from_model_badmodel():
     h = HRSOrder(order=16)
     y = np.arange(25)
     y = y.reshape(5,5)
     h.set_order_from_array(y)
     with pytest.raises(TypeError):
          h.set_wavelength_from_model(5,h.region, wavelength_unit=u.nm)

#add extract_spectrum tests
def test_hrsorder_extract_spectrum_no_wavelength():
    h = create_hrsorder()
    h.wavelength = None
    with pytest.raises(ValueError):
         s = h.extract_spectrum()

def test_hrsorder_extract_spectrum_no_wavelength_unit():
    h = create_hrsorder()
    h.wavelength_unit = None
    with pytest.raises(ValueError):
         s = h.extract_spectrum()

def test_hrsorder_extract_spectrum_no_flux():
    h = create_hrsorder()
    h.flux = None
    with pytest.raises(ValueError):
         s = h.extract_spectrum()

def test_hrsorder_extract_spectrum_no_flux_unit():
    h = create_hrsorder()
    h.flux_unit = None
    with pytest.raises(ValueError):
         s = h.extract_spectrum()

def test_hrsorder_extract_spectrum():
    h = create_hrsorder()
    s = h.extract_spectrum()
    assert len(s.flux) == len(h.flux)


