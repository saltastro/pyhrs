import os
import sys
import numpy as np
import pickle

from ccdproc import CCDData

import specutils
from astropy import units as u

from astropy import modeling as mod
from astropy.io import fits

import pylab as pl

import specreduce

from specreduce.interidentify import InterIdentify
from specreduce import spectools as st
from specreduce import WavelengthSolution

from PyQt4 import QtGui,QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar


from pyhrs import mode_setup_information
from pyhrs import zeropoint_shift
from pyhrs import HRSOrder, HRSModel


import argparse


class FitWavelengthWindow(QtGui.QWidget):

   def __init__(self,l_xarr, l_warr, l_oarr,xarr,farr,oarr,outfile):
      self.l_xarr=l_xarr 
      self.l_warr=l_warr 
      self.l_oarr=l_oarr 
      self.xarr=xarr
      self.farr=farr
      self.oarr=oarr
      self.outfile=outfile

      super(FitWavelengthWindow,self).__init__()
      self.setWindowTitle("Fittig 2D wavelegth solution")
      self.layout()
        
   def layout(self):
        
      self.figure = plt.figure()
      self.canvas = FigureCanvas(self.figure)
      self.toolbar = NavigationToolbar(self.canvas, self)

      self.fit_orderL = QtGui.QLabel('Fit order')
      self.fit_orderE = QtGui.QLineEdit()
      self.fit_orderE.setText("5")

      self.iterationsL = QtGui.QLabel('Iterations')
      self.iterationsE = QtGui.QLineEdit()
      self.iterationsE.setText("10")

      self.lowrejL = QtGui.QLabel('Lower rejection')
      self.lowrejE = QtGui.QLineEdit()
      self.lowrejE.setText("1")

      self.uprejL = QtGui.QLabel('Upper rejection')
      self.uprejE = QtGui.QLineEdit()
      self.uprejE.setText("1")

      self.fitbutton=QtGui.QPushButton("Refit",self)
      self.fitbutton.clicked.connect(self.fit)

      self.exitbutton=QtGui.QPushButton("Exit + Save",self)
      self.exitbutton.clicked.connect(self.exit_fitting)

      grid = QtGui.QGridLayout()
      grid.setSpacing(10)

      grid.addWidget(self.canvas, 1, 0,1,8)
      grid.addWidget(self.toolbar, 2, 0,1,8)

      grid.addWidget(self.fit_orderL, 3, 0)
      grid.addWidget(self.fit_orderE, 3, 1)
      grid.addWidget(self.iterationsL, 3, 2)
      grid.addWidget(self.iterationsE, 3, 3)
      grid.addWidget(self.lowrejL, 3, 4)
      grid.addWidget(self.lowrejE, 3, 5)
      grid.addWidget(self.uprejL, 3, 6)
      grid.addWidget(self.uprejE, 3, 7)

      grid.addWidget(self.fitbutton, 4, 0,1,6)
      grid.addWidget(self.exitbutton, 4, 7,1,1)

        
      self.setLayout(grid)
    
      self.fit()
      self.show()


   def fit(self):

      plt.clf()

      try:                                 #reading parameters of the fit
         fit_order=int(self.fit_orderE.text())
         iterations=int(self.iterationsE.text())
         lowrej=float(self.lowrejE.text())
         uprej=float(self.uprejE.text())
      except ValueError:
         print "Bad input"

      p_init = mod.models.Polynomial2D(degree=fit_order)
      fit_p = mod.fitting.LevMarLSQFitter()

      for i in range(iterations):          #iterative fitting
         if i==0:                          #first iteration - there are no residuals available
            self.p = fit_p(p_init,self.l_xarr, self.l_oarr, self.l_warr)
            i_fit=range(len(self.l_xarr))
        
         else:                             #fitting only to good points (not outliers)
            if i_fit[0].size==0:           #if all point are outliers print error
               print "Error while rejecting points - try higher values of upper rejection and lower rejection"
            self.p = fit_p(p_init,self.l_xarr[i_fit], self.l_oarr[i_fit], self.l_warr[i_fit])



         fitted_lines=self.p(self.l_xarr, self.l_oarr)
         residuals_lines=(self.l_warr-fitted_lines)


         #rejecting outliers
         std_res=np.std(residuals_lines)

         #making list of outliers (only for plotting)
         i_tmp=range(len(self.l_xarr))
         mask=np.in1d(i_tmp, i_fit)
         i_fit=np.where(mask & (residuals_lines<uprej*std_res)  & (residuals_lines>-lowrej*std_res))
        
         
      print "Mean absolute value of residuals: ",np.mean(np.abs(residuals_lines))
      print "Mean absolute value of residuals of lines used for fitting: ",np.mean(np.abs(residuals_lines[i_fit]))
      print "Standard deviation of fit (all lines): ",np.std(residuals_lines)
      print "Standard deviation of fit (lines without otliers): ",np.std(residuals_lines[i_fit])
      print "Total number of arc lines: ",len(self.l_warr)
      print "Number of arc lines after rejecting outliers: ",len(i_fit[0])
      print

      ax1 = self.figure.add_subplot(131)
      lines_plot = ax1.scatter(self.l_xarr, self.l_oarr, c=self.l_warr,edgecolor="None")
      cb1=self.figure.colorbar(lines_plot, orientation='horizontal')
      cb1.set_label(r"Wavelength [$\AA$]")

      ax1.set_ylabel("ORDER")
      ax1.set_xlabel("X [pixel]")
      ax1.hold(False)

      ax2 = self.figure.add_subplot(132, sharex=ax1, sharey=ax1)
      residual1_plot = ax2.scatter(self.l_xarr[i_fit], self.l_oarr[i_fit], c=(self.l_warr[i_fit]-fitted_lines[i_fit]),edgecolor="None")
      cb2=self.figure.colorbar(residual1_plot, orientation='horizontal')
      cb2.set_label(r"Residuals [$\AA$]")

      ax2.set_xlabel("X [pixel]")
      ax2.hold(False)

      ax3 = self.figure.add_subplot(133, sharex=ax1, sharey=ax1)
      residual2_plot = ax3.scatter(self.l_xarr, self.l_oarr, c=(self.l_warr-fitted_lines),edgecolor="None")

      cb2=self.figure.colorbar(residual2_plot, orientation='horizontal')
      cb2.set_label(r"Residuals [$\AA$]")
      ax3.set_xlabel("X [pixel]")
      ax3.hold(False)

      plt.tight_layout()
      self.canvas.draw()


   def exit_fitting(self,textbox):  
                          

      warr=self.p(self.xarr, self.oarr)
      c1 = fits.Column(name='Wavelength', format='D', array=warr, unit='Angstroms')
      c2 = fits.Column(name='Flux', format='D', array=self.farr, unit='Counts')
      c3 = fits.Column(name='Order', format='I', array=self.oarr)

      tbhdu = fits.BinTableHDU.from_columns([c1,c2,c3])
      tbhdu.writeto(self.outfile, clobber=True)
      
      print "Output saved to ", outfile
      self.close()





def read_arclines(ccd, order_frame, soldir):

   l_xarr=np.array([])
   l_warr=np.array([])
   l_oarr=np.array([])
  
   min_order = int(order_frame.data[order_frame.data>0].min())
   max_order = int(order_frame.data[order_frame.data>0].max())
   sp_dict = {}
   for n_order in np.arange(min_order, max_order):
      try:
         shift_dict, ws = pickle.load(open(soldir+'sol_%i.pkl' % n_order))
         l_xarr=np.append(l_xarr,ws.x)
         l_warr=np.append(l_warr,ws.wavelength)
         n_oarr=np.ones_like(ws.x)*n_order
         l_oarr=np.append(l_oarr,n_oarr)
      except IOError: 
         continue
   return l_xarr,l_warr,l_oarr


def xextract_order(ccd, order_frame, n_order,  shift_dict, y1=3, y2=10, target=True, interp=False):
    """Given a wavelength solution and offset, extract the order

    """
    hrs = HRSOrder(n_order)
    hrs.set_order_from_array(order_frame.data)
    hrs.set_flux_from_array(ccd.data, flux_unit=ccd.unit)
    hrs.set_target(target)
    data, coef = hrs.create_box(hrs.flux, interp=interp)

    xarr = np.arange(len(data[0]))
    flux = np.zeros_like(xarr, dtype=float)
    weight = 0
    for i in shift_dict.keys():
        if i < len(data) and i >= y1 and i <= y2:
            m = shift_dict[i]
	    shift_flux = np.interp(xarr, m(xarr), data[i])
            data[i] = shift_flux
            flux += shift_flux * np.median(shift_flux)
            weight += np.median(shift_flux)
    pickle.dump(data, open('box_%i.pkl' % n_order, 'w'))
    return xarr, flux/weight


def read_fits(ccd, order_frame, soldir, interp=False):
   rm, xpos, target, res, w_c, y1, y2 =  mode_setup_information(ccd.header)

   if target=='upper': 
     target=True
   else:
     target=False


   xarr=np.array([])
   farr=np.array([])
   oarr=np.array([])

   min_order = int(order_frame.data[order_frame.data>0].min())
   max_order = int(order_frame.data[order_frame.data>0].max())
    
   for n_order in np.arange(min_order, max_order):
      try:
         shift_dict, ws = pickle.load(open(soldir+'sol_%i.pkl' % n_order))
      except:
         continue
      x, f = xextract_order(ccd, order_frame, n_order, shift_dict, target=target, interp=interp)
      o=np.ones_like(x)*n_order
      xarr=np.append(xarr,x)
      farr=np.append(farr,f)
      oarr=np.append(oarr,o)
   return xarr,farr,oarr




def wave2Dfit(ccd, order_frame, soldir, outfile):

   l_xarr,l_warr,l_oarr=read_arclines(ccd, order_frame, soldir)
   xarr,farr,oarr=read_fits(ccd, order_frame, soldir)

   fitting_app=QtGui.QApplication(sys.argv)
   fitting_window=FitWavelengthWindow(l_xarr,l_warr,l_oarr,xarr,farr,oarr, outfile)
   fitting_window.raise_()
   fitting_app.exec_()
   fitting_app.deleteLater()
   return True


if __name__=='__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("spectrum_fits",help="Fits file with an extracted HRS spectrum",type=str)
   parser.add_argument("order_frame",help="HRS order frame",type=str)
   parser.add_argument("calibration_folder",help="Path to the lr/mr/hr calibration folder",type=str)
   args=parser.parse_args()
  
   ccd = CCDData.read(args.spectrum_fits) 
   order_frame = CCDData.read(args.order_frame, unit=u.adu) 
   soldir = args.calibration_folder

   outfile = (args.spectrum_fits).replace('.fits', '_spec.fits')

   wave2Dfit(ccd, order_frame, soldir, outfile)
