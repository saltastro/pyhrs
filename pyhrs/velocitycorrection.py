from astropy.time import Time
from astropy.coordinates import SkyCoord, solar_system, EarthLocation, ICRS, UnitSphericalRepresentation, CartesianRepresentation
from astropy import units as u
from astropy import constants as c
from astropy import coordinates
from PySpectrograph.Spectra.Spectrum import air2vac

__all__=['velcorr', 'convert_data']

def velcorr(time, skycoord, location=None):
  """Barycentric velocity correction.
  
  Uses the ephemeris set with  ``astropy.coordinates.solar_system_ephemeris.set`` for corrections. 
  For more information see `~astropy.coordinates.solar_system_ephemeris`.
  
  Parameters
  ----------
  time : `~astropy.time.Time`
    The time of observation.
  skycoord: `~astropy.coordinates.SkyCoord`
    The sky location to calculate the correction for.
  location: `~astropy.coordinates.EarthLocation`, optional
    The location of the observatory to calculate the correction for.
    If no location is given, the ``location`` attribute of the Time
    object is used
    
  Returns
  -------
  vel_corr : `~astropy.units.Quantity`
    The velocity correction to convert to Barycentric velocities. Should be added to the original
    velocity.
  """
  
  if location is None:
    if time.location is None:
        raise ValueError('An EarthLocation needs to be set or passed '
                         'in to calculate bary- or heliocentric '
                         'corrections')
    location = time.location
    
  # ensure sky location is ICRS compatible
  if not skycoord.is_transformable_to(ICRS()):
    raise ValueError("Given skycoord is not transformable to the ICRS")
  
  ep, ev = solar_system.get_body_barycentric_posvel('earth', time) # ICRS position and velocity of Earth's geocenter
  op, ov = location.get_gcrs_posvel(time) # GCRS position and velocity of observatory
  # ICRS and GCRS are axes-aligned. Can add the velocities
  velocity = ev + ov # relies on PR5434 being merged
  
  # get unit ICRS vector in direction of SkyCoord
  sc_cartesian = skycoord.icrs.represent_as(UnitSphericalRepresentation).represent_as(CartesianRepresentation)
  return sc_cartesian.dot(velocity).to(u.km/u.s) # similarly requires PR5434

def convert_data(wave, vslr):
   """Convert wavelenght array for vacumm wavlength and to the v_slr 
     
   """
   wave = air2vac(wave)
   return wave * (1+vslr/c.c)
