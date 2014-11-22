# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module tests the HRSModel class
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..hrsorder import HRSOrder


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

#test setting the flux
 
#test setting the wavelength

#test setting the wavelength


