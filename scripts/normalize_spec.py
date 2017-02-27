import sys
from PyQt4 import QtGui,QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from astropy.io import fits
import numpy as np

import argparse

class FitWindow(QtGui.QWidget):
   """A class describing with PyQt4 GUI for normalizing the spectra by fitting continuum .


   Parameters
   ----------
   warr: ~numpy.ndarray
       Array with wavelength

   farr: ~numpy.ndarray
       Array with fluxes

   oarr: ~numpy.ndarray
       Array with orders numbers

   outfile: ~str
       Name of the fits file the output will be written to
 
   fittype: ~str
       Fitting type. Can be "individual" for fitting individual orders. For other values of this parameter whole spectrum will be fit simultaneously.

   Returns
   -------
       The resulting spectrum will be saved to "outfile". The normalization will be performed only for orders which were inspected by hand.

   Notes
   -----
       Rejected region should be in format "(Lower wavelength)-(Higher wavelength)" without spaces. In case of multiple regions 
  
   """
   def __init__(self,warr,farr, oarr,outfile,fittype):
      self.warr=warr #table with wavelengths
      self.farr=farr #table with fluxes
      self.oarr=oarr #table with orders
      self.fittype=fittype #fit type
      self.outfile=outfile #name of output file

      super(FitWindow,self).__init__()
      self.setWindowTitle("Normalizing spectrum")
      self.layout()
        
   def layout(self):
        
      self.figure = plt.figure()
      self.canvas = FigureCanvas(self.figure)
      self.toolbar = NavigationToolbar(self.canvas, self)

      self.fit_orderL = QtGui.QLabel('Fit order')
      self.fit_orderE = QtGui.QLineEdit()
      self.fit_orderE.setText("6")

      self.iterationsL = QtGui.QLabel('Iterations')
      self.iterationsE = QtGui.QLineEdit()
      self.iterationsE.setText("10")

      self.lowrejL = QtGui.QLabel('Lower rejection')
      self.lowrejE = QtGui.QLineEdit()
      self.lowrejE.setText("1")

      self.uprejL = QtGui.QLabel('Upper rejection')
      self.uprejE = QtGui.QLineEdit()
      self.uprejE.setText("3")


      self.rej_regionL = QtGui.QLabel('Rejection regions')
      self.rej_regionE = QtGui.QLineEdit()

      self.fitbutton=QtGui.QPushButton("Refit",self)
      self.fitbutton.clicked.connect(self.fit)

      self.exitbutton=QtGui.QPushButton("Exit + Save",self)
      self.exitbutton.clicked.connect(self.exit_fitting)

      self.nextbutton=QtGui.QPushButton("Next",self)
      self.nextbutton.clicked.connect(self.next_order)


      self.previousbutton=QtGui.QPushButton("Previous",self)
      self.previousbutton.clicked.connect(self.previous_order)

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
      grid.addWidget(self.rej_regionL, 4, 0,1,1)
      grid.addWidget(self.rej_regionE, 4, 1,1,7)


      grid.addWidget(self.fitbutton, 5, 0,1,2)
      grid.addWidget(self.previousbutton, 5, 2,1,2)
      grid.addWidget(self.nextbutton, 5, 4,1,2)
      grid.addWidget(self.exitbutton, 5, 7,1,1)

        
      
      self.orders=np.unique(self.oarr) #list of all orders in the spectrum
      self.order_count=0 #keeps track of witch orders is being normalized

      self.coefficients_tab=np.zeros_like(self.orders).tolist() #array that stores coefficients of polynomial fit to the order

      self.fit()
      self.setLayout(grid)
      self.show()

   def fit(self):                          #fitting part of module
   
      plt.clf()                            #cleaning plots so that previous fits are not stored on screen

      try:                                 #reading parameters of the fit
         fit_order=int(self.fit_orderE.text())
         iterations=int(self.iterationsE.text())
         lowrej=float(self.lowrejE.text())
         uprej=float(self.uprejE.text())
         rej_regions=str(self.rej_regionE.text()).split()
      except ValueError:
         print "Bad input"

     
      if self.fittype=="individual":       #if fittype=="individual" fit will be only to one order
         o=self.orders[self.order_count]
         o=int(o)
         i_order=np.where(self.oarr==o)
      else:                                #fitting to all orders and 
         i_order=np.where(self.farr > 0.)

   


      for i in range(iterations):          #iterative fitting
         if i==0:                          #first iteration - there are no residuals available
            coefficients=np.polyfit(self.warr[i_order], self.farr[i_order], fit_order)
        
         else:                             #fitting only to good points (not outliers)
            if i_fit[0].size==0:           #if all point are outliers print error
               print "Error while rejecting points - try higher values of upper rejection and lower rejection"
               i_fit=order_i
            coefficients=np.polyfit(self.warr[i_fit], self.farr[i_fit], fit_order)
               
         #actual fitting
         func=np.poly1d(coefficients)
         fitted=np.polyval(func,self.warr)
         residuals=(self.farr-fitted)


         #rejecting outliers and bad regions
         mean_res=np.mean(residuals[i_order])
         std_res=np.std(residuals[i_order])
         if self.fittype=="individual":
            i_fit=np.where( (self.oarr==o) & (residuals<uprej*std_res)  & (residuals>-lowrej*std_res))
         else:   
            i_fit=np.where( (self.farr > 0.) & (residuals<uprej*std_res)  & (residuals>-lowrej*std_res))

         for j in range(len(rej_regions)):
            region=np.array(rej_regions[j].split("-")).astype(float)
            if len(region)==2:
               i_tmp=range(len(self.warr))
               i_region=np.where((self.warr > np.min(region)) & (self.warr < np.max(region)))
               mask_r1=np.in1d(i_tmp, i_region)
               mask_r2=np.in1d(i_tmp, i_fit)
               i_fit=np.where(~mask_r1 & mask_r2)
            else:
               print "Bad region: ",rej_regions[j]
            

      #making list of outliers (only for plotting)
      i_outliers=range(len(self.farr))
      mask1=np.in1d(i_outliers, i_fit)
      mask2=np.in1d(i_outliers, i_order)
      i_outliers=np.where(~mask1 & mask2)

      self.coefficients_tab[self.order_count]=coefficients      #storing the fit coefficients 

      ax1 = self.figure.add_subplot(211)
      ax1.plot(self.warr[i_order],self.farr[i_order],c="green") 
      ax1.scatter(self.warr[i_outliers],self.farr[i_outliers],c="red",edgecolor="None") 
      
      ax1.axes.get_xaxis().set_visible(False)
      ax1.plot(self.warr[i_order],fitted[i_order],c="blue")

      ax2 = self.figure.add_subplot(212, sharex=ax1)
      ax2.hold(False)
      ax2.plot(self.warr[i_order],self.farr[i_order]/fitted[i_order],c="blue")
      ax2.set_xlabel(r"Wavelength [$\AA$]")

      plt.tight_layout()
      self.canvas.draw()


   def next_order(self):                                        #moving to the next order when fitting individual orders and the current order is not the last one
      if (not self.order_count==(len(self.orders)-1) ) and (self.fittype=="individual"):
         self.order_count+=1
      self.fit()

   def previous_order(self):                                    #moving to the previous order when fitting individual orders and the current order is not the first one
      if not self.order_count==0:
         self.order_count-=1
      self.fit()

   def exit_fitting(self,textbox):                              #exit fitting window and store the normalized spectrum in outfile
      
      for j in range(len(self.orders)):
         check_fit=False                                        #checks whether fit was performed for this order
         if self.fittype=="individual":                         #if fitting individual orders
            if type(self.coefficients_tab[j]) is np.ndarray:    #if the fitting was performed for this order 
                  func=np.poly1d(self.coefficients_tab[j])
                  check_fit=True
         else:
            func=np.poly1d(self.coefficients_tab[0])            #if fitting all the orders simultaneously the fitting coefficients are always stored in the first
                                                                #element of coefficients_tab
            check_fit=True

         fitted=np.polyval(func,self.warr)
         i_order=np.where(self.oarr==self.orders[j])
         
         if check_fit:
            self.farr[i_order]=self.farr[i_order]/fitted[i_order]
         else:
            self.farr[i_order]=self.farr[i_order]
     
      c1 = fits.Column(name='Wavelength', format='D', array=self.warr, unit='Angstroms')
      c2 = fits.Column(name='Flux', format='D', array=self.farr, unit='Counts')
      c3 = fits.Column(name='Order', format='I', array=self.oarr)

      tbhdu = fits.BinTableHDU.from_columns([c1,c2,c3])
      tbhdu.writeto(outfile, clobber=True)         
      self.close()


def normalize_spec(warr,farr,oarr,outfile,fit="individual"):

   """A function calling the fitting window


   Parameters
   ----------
   warr: ~numpy.ndarray
       Array with wavelength

   farr: ~numpy.ndarray
       Array with fluxes

   oarr: ~numpy.ndarray
       Array with orders numbers

   outfile: ~str
       Name of the fits file the output will be written to
 
   fittype: ~str
       Fitting type. Can be "individual" for fitting individual orders. For other values of this parameter whole spectrum will be fit simultaneously.
  
   """
   fitting_app=QtGui.QApplication(sys.argv)
   fitting_window=FitWindow(warr,farr,oarr,outfile,fit)
   fitting_window.raise_()
   fitting_app.exec_()
   fitting_app.deleteLater()
   return True


if __name__=="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("spectrum_fits",help="Fits file with an extracted HRS spectrum",type=str)
   parser.add_argument("-a","--all",help="Fits all orders simultaneously",action="store_true")
   args=parser.parse_args()

   img_sci = args.spectrum_fits
   hdu_sci = fits.open(img_sci)
   wave_sci = hdu_sci[1].data['Wavelength']
   flux_sci = hdu_sci[1].data['Flux']
   order_sci = hdu_sci[1].data['Order']

   #sorting arrays
   i_sort=np.argsort(wave_sci)
   wave_sci=wave_sci[i_sort]
   flux_sci=flux_sci[i_sort]
   order_sci=order_sci[i_sort]

   outfile="n"+sys.argv[1]
   if args.all:
      normalize_spec(wave_sci,flux_sci,order_sci,outfile,fit="all")
   else:
      normalize_spec(wave_sci,flux_sci,order_sci,outfile)
