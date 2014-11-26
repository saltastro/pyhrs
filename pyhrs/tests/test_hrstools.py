# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module tests the hrstools
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest

from ..hrsmodel import HRSModel
from PySpectrograph import SpectrographError
