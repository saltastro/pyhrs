import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits as pyfits
from sklearn.preprocessing import normalize

from astropy.modeling import models, fitting


#=====================================================================================
def polyfitr(x, y, order, clip, xlim=None, ylim=None, mask=None, debug=False):
    """ Fit a polynomial to data, rejecting outliers.

    Fits a polynomial f(x) to data, x,y.  Finds standard deviation of
    y - f(x) and removes points that differ from f(x) by more than
    clip*stddev, then refits.  This repeats until no points are
    removed.

    Inputs
    ------
    x,y:
        Data points to be fitted.  They must have the same length.
    order: int (2)
        Order of polynomial to be fitted.
    clip: float (6)
        After each iteration data further than this many standard
        deviations away from the fit will be discarded.
    xlim: tuple of maximum and minimum x values, optional
        Data outside these x limits will not be used in the fit.
    ylim: tuple of maximum and minimum y values, optional
        As for xlim, but for y data.
    mask: sequence of pairs, optional
        A list of minimum and maximum x values (e.g. [(3, 4), (8, 9)])
        giving regions to be excluded from the fit.
    debug: boolean, default False
        If True, plots the fit at each iteration in matplotlib.

    Returns
    -------
    coeff, x, y:
        x, y are the data points contributing to the final fit. coeff
        gives the coefficients of the final polynomial fit (use
        np.polyval(coeff,x)).

    Examples
    --------
    >>> x = np.linspace(0,4)
    >>> np.random.seed(13)
    >>> y = x**2 + np.random.randn(50)
    >>> coeff, x1, y1 = polyfitr(x, y)
    >>> np.allclose(coeff, [1.05228393, -0.31855442, 0.4957111])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, order=1, xlim=(0.5,3.5), ylim=(1,10))
    >>> np.allclose(coeff, [3.23959627, -1.81635911])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, mask=[(1, 2), (3, 3.5)])
    >>> np.allclose(coeff, [1.08044631, -0.37032771, 0.42847982])
    True
    """

    x = np.asanyarray(x)
    y = np.asanyarray(y)
    isort = x.argsort()
    x, y = x[isort], y[isort]

    keep = np.ones(len(x), bool)
    if xlim is not None:
        keep &= (xlim[0] < x) & (x < xlim[1])
    if ylim is not None:
        keep &= (ylim[0] < y) & (y < ylim[1])
    if mask is not None:
        badpts = np.zeros(len(x), bool)
        for x0,x1 in mask:
            badpts |=  (x0 < x) & (x < x1)
        keep &= ~badpts

    x,y = x[keep], y[keep]
    if debug:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y,'.')
        ax.set_autoscale_on(0)
        pl.show()

    coeff = np.polyfit(x, y, order)
    if debug:
        pts, = ax.plot(x, y, '.')
        poly, = ax.plot(x, np.polyval(coeff, x), lw=2)
        pl.show()
        raw_input('Enter to continue')
    norm = np.abs(y - np.polyval(coeff, x))
    stdev = np.std(norm)
    condition =  norm < clip * stdev
    y = y[condition]
    x = x[condition]
    while norm.max() > clip * stdev:
        if len(y) < order + 1:
            raise Exception('Too few points left to fit!')
        coeff = np.polyfit(x, y, order)
        if debug:
            pts.set_data(x, y)
            poly.set_data(x, np.polyval(coeff, x))
            pl.show()
            raw_input('Enter to continue')
        norm = np.abs(y - np.polyval(coeff, x))
        stdev = norm.std()
        condition =  norm < clip * stdev
        y = y[condition]
        x = x[condition]

    return coeff,x,y
    
#=====================================================================================
def extract_normalize(spectrum_file, order_to_extract):
	sigma_clip = 3.5
	polyfit_order = 3
	img = spectrum_file
	hdu = pyfits.open(img)
	wave = hdu[1].data['Wavelength']
	flux = hdu[1].data['Flux']
	order = hdu[1].data['Order']
	order_to_plot = order_to_extract
	mask = (order==order_to_plot)
	pyhrs_wavelength = wave[mask].copy()
	pyhrs_fluxlvl = flux[mask].copy()
	## Try to fit a cotinuum to the backgraound and subtract this
	## This makes is much easier to fit the Gaussians
	py_coeff, py_C_Wave, py_C_offsets = polyfitr(pyhrs_wavelength, pyhrs_fluxlvl, order=polyfit_order, clip=sigma_clip)
	py_p = np.poly1d(py_coeff)
	pyhrs_fluxlvl = pyhrs_fluxlvl-np.polyval(py_p, pyhrs_wavelength)
	# First do some normalization on the spectrum
	final_pyhrs_flux = pyhrs_fluxlvl
	final_pyhrs_flux = normalize(final_pyhrs_flux.reshape(1,-1), norm='l2')[0]
	# Make the final plot
	makeplot=False
	if makeplot:
		fig1 = plt.figure()
		plt.plot(pyhrs_wavelength, pyhrs_fluxlvl, color='green', linewidth=1.0, label='Normal PyHRS Flux')
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux')
		plt.legend()
		plt.show()
		fig2 = plt.figure()
		plt.plot(pyhrs_wavelength, final_pyhrs_flux, color='blue', linewidth=1.0, label='Normalized PyHRS Reduction')
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux [normalized]')
		plt.legend()
		plt.show()
		
	return pyhrs_wavelength, final_pyhrs_flux
#=====================================================================================
def get_telluric_lines(wavelength_array):
	import ref_index
	
	line_list_file = '/home/rudi/Data/Alexei/20160418/2016-1-MLT-002/atmabs.fits'
	
	line_data = pyfits.open(line_list_file)
	line_tab = line_data[1].data
	line_data.close()

	Telluric_Line_List_Wavelength_start     = line_tab.field('wavelength_start')
	Telluric_Line_List_Wavelength_end     	= line_tab.field('wavelength_end')
	Telluric_Line_List_Wavelength_intensity = line_tab.field('intensity')
	Telluric_Line_List_Wavelength_centre    = line_tab.field('wavelength_centre')

	Above_Min_Wavelength = (Telluric_Line_List_Wavelength_centre > np.min(wavelength_array))
	Below_Max_Wavelength = (Telluric_Line_List_Wavelength_centre < np.max(wavelength_array))

	telluric_lines_centres = Telluric_Line_List_Wavelength_centre[np.where((Below_Max_Wavelength*Above_Min_Wavelength) == True)]
	Telluric_Lines_Str = Telluric_Line_List_Wavelength_start[np.where((Below_Max_Wavelength*Above_Min_Wavelength) == True)]
	Telluric_Lines_End = Telluric_Line_List_Wavelength_end[np.where((Below_Max_Wavelength*Above_Min_Wavelength) == True)]
	
	telluric_lines_centres = ref_index.vac2air(telluric_lines_centres)
	
	return telluric_lines_centres
		
#=====================================================================================
def do_fitting(wavelength_array, flux_array, telluric_linelist):
	gauss_width = 1.0
	search_width = 1.5
	
	number_of_fits = 0
	
	for i in telluric_linelist:
		around_line = wavelength_array[np.where(np.abs(wavelength_array-i) < search_width)]
		around_line_flux = flux_array[np.where(np.abs(wavelength_array-i) < search_width)]
		amplitude = np.max(around_line_flux) - np.min(around_line_flux)
		mean = np.mean(around_line)
		g_init = models.Gaussian1D(amplitude, mean, stddev=0.5)
		fit_g = fitting.LevMarLSQFitter()
		try:
			g = fit_g(g_init, around_line, around_line_flux-np.median(around_line_flux))
		except:
			print "No Guassian could be fit at this position"
			sys.exit()
		
		#print i, g.mean[0], g.amplitude[0], g.stddev[0], "\n"
		if (g.amplitude[0] < -0.01) and (np.abs(g.stddev[0]) < 0.2) and (np.abs(g.mean[0] - i) < 0.025):
			number_of_fits += 1
			#print i, g.mean[0], g.amplitude[0], g.stddev[0], "\n"
			plot_gaussian_fits = False
			if plot_gaussian_fits:
				fig = plt.figure()
				plt.plot(around_line, around_line_flux, color='green', linewidth=1.0, label='pyHRS Reduction')
				plt.plot(around_line, g(around_line), color='magenta', linewidth=1.0, label="Gaussian Fit")
				ymin, ymax = plt.ylim()
				plt.vlines(i, ymin=ymin, ymax=ymax, linestyle='--', linewidth=1.0, color='black', label="Telluric Lines Centre")
				plt.vlines(g.mean[0], ymin=ymin, ymax=ymax, linestyle='--', linewidth=1.0, color='magenta', label="Gaussian Fit Centre")
				plt.xlabel('Wavelength (A)')
				plt.ylabel('Flux [normalized]')
				plt.legend()
				plt.show()
			
	if number_of_fits > 2:
		good_fit = True
	else:
		good_fit = False
	return good_fit, number_of_fits
	
#=====================================================================================	
if __name__=='__main__':
	'''USAGE for RED : telluric_fitter.py raw/pR201604180019_spec.fits 71
	'''
	spectrum_file = sys.argv[1]
	order_to_extract = int(sys.argv[2])
	
	pyhrs_wavelength, pyhrs_norm_flux = extract_normalize(spectrum_file, order_to_extract)
		
	pyhrs_norm_flux = np.roll(pyhrs_norm_flux, 9) # Shift for red images
	
	telluric_lines = get_telluric_lines(pyhrs_wavelength)
	
	fig = plt.figure()
	plt.plot(pyhrs_wavelength, pyhrs_norm_flux, color='green', linewidth=1.0, label='Normalized PyHRS Flux')
	ymin, ymax = plt.ylim()
	plt.vlines(telluric_lines, ymin=ymin, ymax=ymax, linestyle='--', linewidth=1.0, color='black', label="Telluric Lines")
	plt.xlabel('Wavelength (A)')
	plt.ylabel('Flux [normalized]')
	plt.legend()
	plt.show()

	goodfit, lines_fitted = do_fitting(pyhrs_wavelength, pyhrs_norm_flux, telluric_lines)
	
	if goodfit:
		print "There is a good fit between the telluric features and the extracted spectrum"
	else:
		print "No good fit between the telluric features and the extracted spectrum could be found"
	
	
