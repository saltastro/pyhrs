
import sys
from astropy.io import fits
import numpy as np
from pylab import *



if __name__=='__main__':


   img_sci = sys.argv[1]
   hdu_sci = fits.open(img_sci)
   wave_sci = hdu_sci[1].data['Wavelength']
   flux_sci = hdu_sci[1].data['Flux']
   order_sci = hdu_sci[1].data['Order']

   #sorting arrays to have 
   i_sort=np.argsort(wave_sci)
   wave_sci=wave_sci[i_sort]
   flux_sci=flux_sci[i_sort]
   order_sci=order_sci[i_sort]


   if 'individual' in sys.argv:
      w_arr=np.array([])
      f_arr=np.array([])
      o_arr=np.array([])

      for o in np.unique(order_sci):
         o = int(o)

         order_i=np.where(order_sci==o )

         print 
         print "Normalizing order ",o

         check=True
         while check:
            check_input=True
            while check_input:
               check_input=False
               fit_order = raw_input("Order of the fit (default = 3): ")
               if fit_order=="":
                  fit_order=3
               else:
                  try:
                     fit_order=int(fit_order)
                  except ValueError:
                     print "incorrect input"
                     check_input=True

            check_input=True
            while check_input:
               check_input=False
               iterations = raw_input("Number of iterations (default = 15): ")
               if iterations=="":
                  iterations=15
               else:
                  try:
                     iterations=int(iterations)
                  except ValueError:
                     print "incorrect input"
                     check_input=True

            check_input=True
            while check_input:
               check_input=False
               lowrej = raw_input("Lower rejection limit (default = 0.0001): ")
               if lowrej=="":
                  lowrej=0.0001
               else:
                  try:
                     lowrej=float(lowrej)
                  except ValueError:
                     print "incorrect input"
                     check_input=True

            check_input=True
            while check_input:
               check_input=False
               uprej = raw_input("Upper rejection limit (default = 3): ")
               if uprej=="":
                  uprej=3
               else:
                  try:
                     uprej=float(uprej)
                  except ValueError:
                     print "incorrect input"
                     check_input=True
           
            for i in range(iterations):
               if i==0:
                  coefficients=np.polyfit(wave_sci[order_i], flux_sci[order_i], fit_order)
               else:
                  if i_fit[0].size==0:
                     print "error while rejecting point - try higher values of uprej and lowrej"
                     i_fit=order_i
                  coefficients=np.polyfit(wave_sci[i_fit], flux_sci[i_fit], fit_order)

               func=np.poly1d(coefficients)
               fitted=np.polyval(func,wave_sci)
               residuals=(flux_sci-fitted)

               mean_res=np.mean(residuals)
               std_res=np.std(residuals)
               i_fit=np.where( (order_sci==o)  & (residuals<uprej*std_res)  & (residuals>-lowrej*std_res))


            figure()
            subplot(2,1, 1)
            plot(wave_sci[order_i], flux_sci[order_i])
            plot(wave_sci[order_i], fitted[order_i])
            ylabel("Input spectrum ")
            subplot(2, 1, 2)
            plot(wave_sci[order_i], flux_sci[order_i]/fitted[order_i])
            xlabel(r"Wavelength [$\AA$]")
            ylabel("Normalized spectrum ")
            show()



            check_input=True
            while check_input==True:
               repeat=raw_input("Repeat fitting? (default = no): ")
               if repeat=="y" or repeat=="yes" or repeat=="Y" or repeat=="YES":
                  check_input=False
               elif repeat=="n" or repeat=="no" or repeat=="N" or repeat=="NO" or repeat=="":
                  check_input=False
                  check=False
            print
         w_arr=np.append(w_arr,wave_sci[order_i])
         f_arr=np.append(f_arr,flux_sci[order_i]/fitted[order_i])
         o_arr=np.append(o_arr,order_sci[order_i])
    
      c1 = fits.Column(name='Wavelength', format='D', array=w_arr, unit='Angstroms')
      c2 = fits.Column(name='Flux', format='D', array=f_arr, unit='Counts')
      c3 = fits.Column(name='Order', format='I', array=o_arr)

      outfile="n"+sys.argv[1]
      tbhdu = fits.BinTableHDU.from_columns([c1,c2,c3])
      tbhdu.writeto(outfile, clobber=True)
        
   elif 'all' in sys.argv:
    


      check=True
      while check:
         check_input=True
         while check_input:
            check_input=False
            fit_order = raw_input("Order of the fit (default = 6): ")
            if fit_order=="":
               fit_order=6
            else:
               try:
                  fit_order=int(fit_order)
               except ValueError:
                  print "incorrect input"
                  check_input=True

         check_input=True
         while check_input:
            check_input=False
            iterations = raw_input("Number of iterations (default = 5): ")
            if iterations=="":
               iterations=5
            else:
               try:
                  iterations=int(iterations)
               except ValueError:
                  print "incorrect input"
                  check_input=True

         check_input=True
         while check_input:
            check_input=False
            lowrej = raw_input("Lower rejection limit (default = 3): ")
            if lowrej=="":
               lowrej=3
            else:
               try:
                  lowrej=float(lowrej)
               except ValueError:
                  print "incorrect input"
                  check_input=True

         check_input=True
         while check_input:
            check_input=False
            uprej = raw_input("Upper rejection limit (default = 3): ")
            if uprej=="":
               uprej=3
            else:
               try:
                  uprej=float(uprej)
               except ValueError:
                  print "incorrect input"
                  check_input=True
        
         for i in range(iterations):
            if i==0:
               coefficients=np.polyfit(wave_sci, flux_sci, fit_order)
            else:
               if i_fit[0].size==0:
                  print "error while rejecting point - try higher values of uprej and lowrej"
                  i_fit=np.arange(wave_sci)
               coefficients=np.polyfit(wave_sci[i_fit], flux_sci[i_fit], fit_order)

            func=np.poly1d(coefficients)
            fitted=np.polyval(func,wave_sci)
            residuals=(flux_sci-fitted)

            mean_res=np.mean(residuals)
            std_res=np.std(residuals)
            i_fit=np.where(  (residuals<uprej*std_res)  & (residuals>-lowrej*std_res))

         figure()
         subplot(2,1, 1)
         plot(wave_sci, flux_sci)
         plot(wave_sci, fitted)
         ylabel("Input spectrum ")
         subplot(2, 1, 2)
         plot(wave_sci, flux_sci/fitted)
         xlabel(r"Wavelength [$\AA$]")
         ylabel("Normalized spectrum ")
         show()

         check_input=True
         while check_input==True:
            repeat=raw_input("Repeat fitting? (default = no): ")
            if repeat=="y" or repeat=="yes" or repeat=="Y" or repeat=="YES":
               check_input=False
            elif repeat=="n" or repeat=="no" or repeat=="N" or repeat=="NO" or repeat=="":
               check_input=False
               check=False
         print



      flux_sci=flux_sci/fitted
      c1 = fits.Column(name='Wavelength', format='D', array=wave_sci, unit='Angstroms')
      c2 = fits.Column(name='Flux', format='D', array=flux_sci, unit='Counts')
      c3 = fits.Column(name='Order', format='I', array=order_sci)

      outfile="n"+sys.argv[1]
      tbhdu = fits.BinTableHDU.from_columns([c1,c2,c3])
      tbhdu.writeto(outfile, clobber=True)
   else:
      print "Incorrect input"
