# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module tests the HRSModel class
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..hrsmodel import HRSModel
from PySpectrograph import SpectrographError



def test_hrsmodel_empty():
    h = HRSModel()

#test cameras
@pytest.mark.parametrize('camera', ['hrdet', 'hbdet'])
def test_hrsmodel_hrdet_camera(camera):
    h = HRSModel(camera_name=camera)

def test_hrsmodel_wrong_camera():
    with pytest.raises(SpectrographError):
        h = HRSModel(camera_name='badcam')

#test detector
@pytest.mark.parametrize('camera', ['hrdet', 'hbdet'])
def test_hrsmodel_hrdet_detector(camera):
    h = HRSModel(camera_name=camera)

def test_hrsmodel_wrong_detector():
    with pytest.raises(SpectrographError):
        h = HRSModel(camera_name='badcam')

#grating 
@pytest.mark.parametrize('grating', ['hrs', 'red beam', 'blue beam'])
def test_hrsmodel_hrdet_grating(grating):
    h = HRSModel(grating_name=grating)

def test_hrsmodel_wrong_detector():
    with pytest.raises(SpectrographError):
        h = HRSModel(grating_name='badcam')


#test setting the slit
def test_hrsmodel_set_slit():
    h = HRSModel()
    h.set_slit(slitang=2.2)
    assert abs(h.slit.width-0.4927) < 0.001

#test set collimator
def test_hrsmodel_collimator():
    h = HRSModel()
    h.set_collimator('hrs')

def test_hrsmodel_wrong_collimator():
    with pytest.raises(SpectrographError):
        h = HRSModel()
        h.set_collimator('badcollimator')

#test set telescoper
def test_hrsmodel_telescope():
    h = HRSModel()
    h.set_telescope('SALT')

def test_hrsmodel_wrong_collimator():
    with pytest.raises(SpectrographError):
        h = HRSModel()
        h.set_telescope('badcollimator')


#test alpha
def test_hrsmodel_alpha():
    h = HRSModel()
    assert h.alpha() == (h.grating.blaze+h.gamma)

#test beta
def test_hrsmodel_beta():
    h = HRSModel()
    assert h.beta(db=0) == (h.grating.blaze-h.gamma)

#test get wavelength
def test_hrsmodel_wavelength():
    h = HRSModel()
    xarr = 10
    assert abs(1e7*h.get_wavelength(xarr, gamma=h.gamma) - 5665.93) < 0.01

def test_hrsmodel_wavelength_array():
    h = HRSModel()
    xarr = np.array([10.,11.])
    data = 1e7 * h.get_wavelength(xarr, gamma=h.gamma)
    assert abs(data-np.array([5665.93018879, 5665.90777466])).mean() < 0.01
