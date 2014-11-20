# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module tests the HRSModel class
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ccdproc import CCDData
from astropy import units as u
from astropy.utils import NumpyRNGContext



from ..hrsprocess import hrs_process

def create_ccd(size=50, scale=1.0, mean=0.0, seed=123):
    """Create a fake ccd for data testing data processing
    """
    with NumpyRNGContext(seed):
        data = np.random.normal(loc=mean, size=[size, size], scale=scale)

    ccd = CCDData(data, unit=u.adu)
    return ccd


def test_hrs_process_none():
     ccd = create_ccd()
     nccd = hrs_process(ccd)
     assert_array_equal(ccd.data, nccd.data)


#test the overscan subtraction
def test_hrs_process_oscan_region():
    ccd = create_ccd(mean=10.0)
    nccd = hrs_process(ccd, oscan=ccd[:, 1:10])
    assert abs((nccd.data-ccd.data+10.0).mean()) < 0.1

def test_hrs_process_oscan_str():
    ccd = create_ccd(mean=10.0)
    nccd = hrs_process(ccd, oscan='[1:10, 1:50]')
    assert abs((nccd.data-ccd.data+10.0).mean()) < 0.1

def test_hrs_process_oscan_typeerror():
    ccd = create_ccd()
    with pytest.raises(TypeError):
        nccd = hrs_process(ccd, oscan=['aldhfj'])
 
#test the trimming
def test_hrs_process_trim():
    ccd = create_ccd()
    nccd = hrs_process(ccd, trim='[1:40, 1:40]')
    assert nccd.shape == (40,40)

def test_hrs_process_trim_typeerror():
    ccd = create_ccd()
    with pytest.raises(TypeError):
        nccd = hrs_process(ccd, trim=['aldhfj'])

#test subtracting master bias
def test_hrs_process_masterbias_ccddata():
    ccd = create_ccd(mean=10.0)
    bias = create_ccd(mean=9.0)
    nccd = hrs_process(ccd, masterbias=bias)
    assert (nccd.data-1.0).mean() < 0.1

#test subtracting master bias
def test_hrs_process_masterbias_ndarray():
    ccd = create_ccd(mean=10.0)
    bias = create_ccd(mean=9.0)
    bias = bias.data
    nccd = hrs_process(ccd, masterbias=bias)
    assert (nccd.data-1.0).mean() < 0.1


def test_hrs_process_masterbias_typeerror():
    ccd = create_ccd()
    with pytest.raises(TypeError):
        nccd = hrs_process(ccd, masterbias=['aldhfj'])

#test creating the error frame
def test_hrs_process_masterbias_error_nogain():
    ccd = create_ccd()
    nccd = hrs_process(ccd, error=True, gain=2.0, rdnoise=5.0)
    assert nccd.uncertainty is not None


def test_hrs_process_masterbias_error_nogain():
    ccd = create_ccd()
    with pytest.raises(ValueError):
        nccd = hrs_process(ccd, error=True, gain=None, rdnoise=5.0)

def test_hrs_process_masterbias_error_nordnoise():
    ccd = create_ccd()
    with pytest.raises(ValueError):
        nccd = hrs_process(ccd, error=True, gain=1.0, rdnoise=None)

#check the masking
def test_hrs_process_masterbias_error_nogain():
    ccd = create_ccd()
    mask = ccd.data*0.0 
    nccd = hrs_process(ccd, bad_pixel_mask=mask)
    assert nccd.mask is not None
    assert nccd.mask.shape == ccd.data.shape

def test_hrs_process_bad_pixel_mask_typeerror():
    ccd = create_ccd()
    with pytest.raises(TypeError):
        nccd = hrs_process(ccd, bad_pixel_mask=['aldhfj'])

#check the gain
def test_hrs_process_gain():
    ccd = create_ccd(mean=10.0)
    nccd = hrs_process(ccd, gain=2 * u.electron/u.adu)
    assert abs((nccd.data-2*10.0).mean()) < 0.1
    assert nccd.unit == u.electron

def test_hrs_process_gain_typeerror():
    ccd = create_ccd()
    with pytest.raises(TypeError):
        nccd = hrs_process(ccd, oscan=2)

