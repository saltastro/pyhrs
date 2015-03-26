# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module tests the hrstools
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..hrstools import *


def test_background(loc=100.0, scale=10):
    b_arr = np.random.normal(size=(100,100), loc=loc, scale=scale)
    med, mad = background(b_arr)
    assert abs(loc - med) < 1.0
    assert abs(scale - mad) < 1.0

