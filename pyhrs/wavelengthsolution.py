
import numpy as np

from astropy import modeling as md
from astropy import stats

__all__=['WavelengthSolution2D']


class WavelengthSolution2D(object):

    """A class describing the solution between x-position, order, and wavelength.


    Parameters
    ----------
    x: ~numpy.ndarray
        Array of the x-positions  

    order: ~numpy.ndarray
        Array of the order-positions  

    wavelength: ~numpy.ndarray
        Array of the wavelength at each x-position
 
    model: ~astropy.modeling.models
        A 2D model describing the transformation between x, order, and wavelength

    Raises
    ------

    Notes
    -----
 
    Examples
    --------
  
    """
    


    def __init__(self, x, order, wavelength, model):
        self.x = x
        self.order = order
        self.wavelength = wavelength
        self.model = model
        self.mask = (x>0) 

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        #TODO: Add checker that it is an astropy model
        self._model = value

    @property
    def coef(self):
        return self.model.parameters

    @coef.setter
    def coef(self, value):
        self._model.parameters = value

    def __call__(self, x, o):
        return self.model(x, o)

    def fit(self, niter=5, sigma=3,fitter=md.fitting.LinearLSQFitter()):
        """Determine the fit of the model to the data points with rejection

        For each iteraction, a weight is calculated based on the distance a source
        is from the relationship and outliers are rejected.

        Parameters
        ----------
        niter: int
            Number of iteractions for the fit

        sigma: float 
            Sigma rejection factor
 
        fitter: ~astropy.modeling.fitting
            Method for fitting relationship


        """
        weights = np.ones_like(self.x)
        for i in range(niter):
            #self.model = fitter(self.model, self.x, self.order, self.wavelength, weights=weights)
            self.model = fitter(self.model, self.x[self.mask], self.order[self.mask], self.wavelength[self.mask])

            #caculate the weights based on the median absolute deviation
            std = np.std(self.wavelength[self.mask]-self.model(self.x[self.mask], self.order[self.mask]))
            self.mask = (abs(self.wavelength-self.model(self.x, self.order)) < sigma * std)


    def sigma(self, x, o, w):
        """Return the RMS of the fit 
       
        Parameters
        ----------
        x: ~numpy.ndarray
            Array of the x-positions  
        o: ~numpy.ndarray
            Array of the order-positions  
        w: ~numpy.ndarray
            Array of the wavelength-positions  

        Returns
        -------
        
        """
        # if there aren't many data points return the RMS
        if len(x) < 4:
            sigma = (((w - self(x, o)) ** 2).mean()) ** 0.5
        # Otherwise get the average distance between the 16th and
        # 84th percentiles
        # of the residuals divided by 2
        # This should be less sensitive to outliers
        else:
            # Sort the residuals
            rsdls = np.sort(w - self(x,o))
            # Get the correct indices and take their difference
            sigma = (rsdls[int(0.84 * len(rsdls))] -
                   rsdls[int(0.16 * len(rsdls))]) / 2.0
        return sigma

    def chisq(self, x, y, err):
        """Return the chi^2 of the fit"""
        return (((y - self.value(x,o)) / err) ** 2).sum()
