import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['legend.numpoints'] = 1

from astropy.io import fits as fits
from astropy.table import Table
from astropy.modeling import models, fitting

import argparse

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
def extract_normalize(spectrum_file, order_to_extract, polyorder=3, sigmaclip=3.5, makeplot=False):
	""" This will extract a particular order from the reduced PyHRS spectrum files.
	
	Given a reduced PyHRS spectrum files (once data reduction has been perfomed)
	and an order to extract, this routine will try and fit a polynomial to the continuum
	and subtract it from the extracted order.
	
	Inputs
	------
	spectrum_file: string
		The reduced PyHRS spectrum filename.
	order_to_extract: int(3)
		The order number to extract.
	polyorder: int(2)
		The degree of the polynomial when fitting for the continuum.
	sigmaclip: float
		The sigma level to use when rejecting outlier datapoints in the
		polynomial fit.
	makeplot: boolean, default is False
		if True, plots the extract non-continuum corrected flux and the 
		contiuum corrected flux on two seperate figures.
	
	Returns
	------
	pyhrs_wavelength, final_pyhrs_flux:
		The wavelength and continuum correct flux of the extracted order
		is returned.
	
	Examples
	------
	>>>spectrum_file = pR201604180019_spec.fits
	>>>order_to_extract = 71
	>>>pyhrs_wavelength, pyhrs_norm_flux = extract_normalize(spectrum_file, order_to_extract, polyorder=3, sigmaclip=3.0, makeplot=False)
	
	"""
	
	sigma_clip = sigmaclip
	polyfit_order = polyorder
	img = spectrum_file
	hdu = fits.open(img)
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
	xs = np.arange(min(pyhrs_wavelength), max(pyhrs_wavelength), 0.1)
	ys = np.polyval(py_p, xs)
	if makeplot:
		fig1 = plt.figure()
		plt.plot(pyhrs_wavelength, pyhrs_fluxlvl, color='green', linewidth=1.0, label='PyHRS Flux')
		plt.plot(xs, ys, color='red', label="Best Fit with outlier rejection")
		plt.plot(py_C_Wave, py_C_offsets, marker='*', color='red', linestyle="none", label="Points used in fit")
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux')
		plt.legend(scatterpoints=1)
		plt.show()

	pyhrs_fluxlvl = pyhrs_fluxlvl-np.polyval(py_p, pyhrs_wavelength)
	# First do some normalization on the spectrum
	final_pyhrs_flux = pyhrs_fluxlvl
	#final_pyhrs_flux = normalize(final_pyhrs_flux.reshape(1,-1), norm='l2')[0]

	if makeplot:
		fig2 = plt.figure()
		plt.plot(pyhrs_wavelength, final_pyhrs_flux, color='blue', linewidth=1.0, label='Continuum Corrected PyHRS Flux')
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux [normalized]')
		plt.legend(scatterpoints=1)
		plt.show()
		
	return pyhrs_wavelength, final_pyhrs_flux
#=====================================================================================
def get_telluric_lines(wavelength_array, telluric_linelist_file, vacuum=True):
	""" This will load the telluric line list
	
	Reading the telluric line list using astropy Table function
	
	Inputs
	------
	wavelength_array: array
		This is the wavelength array that you want checked against the 
		given telluric line list
		
	telluric_linelist_file: string
		This is the location of the telluric line list file you would like to use.
	
	vacuum: Boolean (default is True)
		Is the telleric line list given in vacuum. Most line lists I've
		seen are provided in vacuum, but it might not always be the case.
	
	Returns
	------
	telluric_lines_centres: array
		The array containing the telluric lines
	
	Examples
	------
	
	"""
	
	import ref_index
	
	line_list_file = telluric_linelist_file
	
	data = Table.read(line_list_file)
	Telluric_Line_List_Wavelength_centre = data['wavelength']

	# Trim the telluric line list to only be within the same
	# wavelength range as the input waveleangth array
	# This saves a lot of time when trying to fit Gaussians
	Above_Min_Wavelength = (Telluric_Line_List_Wavelength_centre > np.min(wavelength_array))
	Below_Max_Wavelength = (Telluric_Line_List_Wavelength_centre < np.max(wavelength_array))

	telluric_lines_centres = Telluric_Line_List_Wavelength_centre[np.where((Below_Max_Wavelength*Above_Min_Wavelength) == True)]
	
	if vacuum:
		telluric_lines_centres = ref_index.vac2air(telluric_lines_centres)
	else:
		pass
	
	return telluric_lines_centres
		
#=====================================================================================
def check_wavelength_solution(wavelength_array, flux_array, telluric_linelist, makeplot=False):
	""" This will check the wavelength solution.
	
	This will check the wavelength solution against the supplied telluric line list
	and return a stutus of GOOD or BAD fit. At each element in the line list
	a Gaussian profile fit is attempted. If a Gaussian could be fit at that 
	position, a counter is iterated. After all elemnts in the line list
	has been checked, the wavelength solution is declared "good" if more than 2
	Gaussian profiles were successully fitted.
	
	Inputs
	------
	wavelength_array: array
		This is the wavelength array that needs to be check against the line list
		to detemine if the wavelength solution is correct.
	flux_array: array (has to be same length as wavelength array)
		This is the associated flux of the wavelength array.
	telluric_linelist:
		An array that contains the wavelengths of all the telluric absorbtion
		lines that need to be checked.
	makeplot: binary (default is False)
		If True, a plot of the succesfully fitted Gaussian profile at each 
		position is shown.
		
	Returns
	------
	good_fit: boolean
		Returns True if more than 2 Gaussian profiles were succesfully fitted.
	
	Examples
	------
	>>
	
	"""
	gauss_width = 1.0
	search_width = 1.5
	
	number_of_fits = 0
	
	lines_fitted = []
	
	if len(telluric_linelist) < 1:
		print "\nNo telluric lines are available for this part of the input wavelength range."
		sys.exit()
	
	for i in telluric_linelist:
		around_line = wavelength_array[np.where(np.abs(wavelength_array-i) < search_width)]
		around_line_flux = flux_array[np.where(np.abs(wavelength_array-i) < search_width)]
		if len(around_line) < 1:
			continue
		if len(around_line_flux) < 1:
			continue
		amplitude = np.max(around_line_flux) - np.min(around_line_flux)
		mean = np.mean(around_line)
		g_init = models.Gaussian1D(amplitude, mean, stddev=0.5)
		fit_g = fitting.LevMarLSQFitter()
		try:
			g = fit_g(g_init, around_line, around_line_flux-np.median(around_line_flux))

			#print i, g.mean[0], g.amplitude[0], g.stddev[0], "\n"
			if (g.amplitude[0] < -0.01) and (np.abs(g.stddev[0]) < 0.2) and (np.abs(g.mean[0] - i) < 0.025):
				number_of_fits += 1
				#print i, g.mean[0], g.amplitude[0], g.stddev[0], "\n"
				plot_gaussian_fits = makeplot
				lines_fitted.append(i)
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

		except:
			print "\nNo Guassian could be fit at this position"
			sys.exit()
		
	if number_of_fits >= 1:
		good_fit = True
	else:
		good_fit = False
	return good_fit, number_of_fits, lines_fitted
	

if __name__=='__main__':
	'''USAGE for RED : telluric_fitter.py raw/pR201604180019_spec.fits 71
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("filename", help="Provide the filename of the reduced pyHRS file you would like to extraxt the spectrum from.")
	parser.add_argument("order_number", help="Number of the order you would ike to extraxt.")
	args = parser.parse_args()
	
	spectrum_file = args.filename
	order_to_extract = int(args.order_number)
	
	telluric_filename = os.path.dirname(os.path.abspath(__file__))+'/tellurics.fits'
	
	## Determine from which arm this image is taken
	Blue_Image = False
	Red_Image = False

	if spectrum_file.find("pH201") != -1:
		Blue_Image = True
		Red_Image = False
		sys.stderr.write("\nFrame from BLUE arm found ...")
		
	if spectrum_file.find("pR201") != -1:
		Blue_Image = False
		Red_Image = True
		sys.stderr.write("\nFrame from RED arm found ...")
	
	
	if Red_Image and (order_to_extract > 81 or order_to_extract < 53):
		print "\nThe input order value in invalid"
		sys.exit()
	if Blue_Image and (order_to_extract < 84 or order_to_extract > 120):
		print "\nThe input order value in invalid"
		sys.exit()
	
	pyhrs_wavelength, pyhrs_norm_flux = extract_normalize(spectrum_file, order_to_extract, makeplot=False)
		
	pyhrs_norm_flux = np.roll(pyhrs_norm_flux, 9) # Shift for red images
	
	telluric_lines = get_telluric_lines(pyhrs_wavelength, telluric_filename, vacuum=True)
	
	goodfit, num_lines_fitted, lines_fitted = check_wavelength_solution(pyhrs_wavelength, pyhrs_norm_flux, telluric_lines, makeplot=False)
	
	if goodfit:
		print "\nThere is a good fit between the telluric features and the extracted spectrum"
	else:
		print "\nNo good fit between the telluric features and the extracted spectrum could be found"
		
	makeplot = False
	if makeplot:
		fig = plt.figure()
		plt.plot(pyhrs_wavelength, pyhrs_norm_flux, color='green', linewidth=1.0, label='Order Corrected PyHRS Flux')
		ymin, ymax = plt.ylim()
		plt.vlines(telluric_lines, ymin=ymin, ymax=ymax, linestyle='--', linewidth=1.0, color='black', label="Telluric Lines")
		plt.vlines(lines_fitted, ymin=ymin, ymax=ymax, linestyle='--', linewidth=1.0, color='red', label="Telluric Lines")
		plt.xlabel('Wavelength (A)')
		plt.ylabel('Flux [normalized]')
		plt.legend(scatterpoints=1)
		plt.show()
	
	
