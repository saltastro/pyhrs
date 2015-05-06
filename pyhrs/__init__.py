# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
pyhrs is a package for reducing data from the High Resolution Spectrograph
on the Southern African Large Telescope
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from hrsmodel import *
    from hrsprocess import *
    from hrsorder import *
    from calibrationframes import *
    from hrstools import *
    from extraction import *
