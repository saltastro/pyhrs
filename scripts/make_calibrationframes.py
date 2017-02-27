#script was heavily based on ipython notebook created by Steve Crawford

import glob
import os
import numpy as np

from astropy.io import fits
from astropy import units as u

from ccdproc import CCDData

from pyhrs import create_masterbias
from pyhrs import create_masterflat
from pyhrs import normalize_image
from pyhrs import create_orderframe

from scipy import ndimage as nd
from astropy import modeling as mod


import argparse
import gc



def create_list_folder():
   """Creates a list of HRS files that are in the current directory

   Parameters
   ----------
    

   Returns
   -------
   blue_files: list of str
       List of HRS blue frames in the running directory
   red_files: list of str
       List of HRS red frames in the running directory

   """
   path=os.getcwd()+os.sep #get current directory
   blue_files = glob.glob('%sH201*.fits'%path)
   red_files = glob.glob('%sR201*.fits'%path)
   return blue_files,red_files



def divide_list(list_file):
   """Checks if every element in provided list is a HRS image and divides the original list into red and blue frames lists

   Parameters
   ----------

   list_file: list of str
       List containing names of HRS files   

   Returns
   -------
   blue_files: list of str
       List of HRS blue frames in the running directory
   red_files: list of str
       List of HRS red frames in the running directory
   """

   blue_files=[]
   red_files=[]

   f=open(list_file,"r")
   for line in f.readlines():
      line_s=line.replace("\n", "")
      if (line_s[-5:]==".fits") and ("H201" in line_s) and not("pH201" in line_s):
         blue_files.append(line_s)
      elif (line_s[-5:]==".fits") and ("R201" in line_s) and not("pR201" in line_s):
         red_files.append(line_s)
      else:
         print("This does not look like a name of a raw HRS file: %s"%line_s)
   f.close()

   return blue_files,red_files






   
def write_bias_flat_arc(files):
   """Creates masterbias masterflat and masterarc files

   Parameters
   ----------

   files: list of str
       List containing names of HRS files   

   Returns
   -------
      Writes HBIAS.fits RBIAS.fits HFLAT.fits RFLAT.fits HARC.fits and RARC.fits in the current directory
   """

   #create a list of calibration frames
   hbias_list = []
   rbias_list = []
   hflat_list = []
   rflat_list = []
   harc_list = []
   rarc_list = []

   for img in files:
      data, header = fits.getdata(img, header=True)
      if not ('DATASEC' in header): 
         #just adding 'DATASEC' keyword if missing      
         header['DATASEC']=header['AMPSEC']
         fits.writeto(img, data, header, clobber=True)
      if header['OBSTYPE']=='Bias' and header['DETNAM']=='HBDET': 
         hbias_list.append(img)
      if header['OBSTYPE']=='Bias' and header['DETNAM']=='HRDET': 
         rbias_list.append(img)
      if 'EXPTYPE' in header:
         if header['EXPTYPE']=='Flat field' and header['DETNAM']=='HBDET': 
            hflat_list.append(img)
         if header['EXPTYPE']=='Flat field' and header['DETNAM']=='HRDET': 
            rflat_list.append(img)
         if header['CCDTYPE']=='Arc' and header['DETNAM']=='HBDET': 
            harc_list.append(img)
         if header['CCDTYPE']=='Arc' and header['DETNAM']=='HRDET': 
            rarc_list.append(img)

      else:
         if header['CCDTYPE']=='Flat field' and header['DETNAM']=='HBDET': 
            hflat_list.append(img)
         if header['CCDTYPE']=='Flat field' and header['DETNAM']=='HRDET': 
            rflat_list.append(img)
         if header['CCDTYPE']=='Arc' and header['DETNAM']=='HBDET': 
            harc_list.append(img)
         if header['CCDTYPE']=='Arc' and header['DETNAM']=='HRDET': 
            rarc_list.append(img)


   if not(hbias_list==[]):
      master_bluebias = create_masterbias(hbias_list)
      master_bluebias.write('HBIAS.fits', clobber=True)
      print("\nCreated blue masterbias\n")
      #this step is just inserted for debugging purposes
      master_bluebias = CCDData.read('HBIAS.fits', ignore_missing_end=True)
      gc.collect()
   if not(hflat_list==[]):
      master_blueflat = create_masterflat(hflat_list) #if bias was created it should probably be "create_masterflat(hflat_list,masterbias=master_bluebias)", but it generates an error
      master_blueflat.write('HFLAT.fits', clobber=True)
      del master_blueflat
      gc.collect()
      print("\nCreated blue masterflat\n")
   if not(harc_list==[]): 
      master_bluearc = create_masterflat(harc_list) #if bias was created it should probably be  "create_masterflat(hflat_list,masterbias=master_bluebias)", but it generates an error
      master_bluearc.write('HARC.fits', clobber=True)
      del master_bluearc
      gc.collect()
      print("\nCreated blue masterarc\n")
   del master_bluebias 
   gc.collect()

   if not(rbias_list==[]):
      master_redbias = create_masterbias(rbias_list)
      master_redbias.write('RBIAS.fits', clobber=True)
      print("\nCreated red masterbias\n")
      #this step is just inserted for debugging purposes
      master_redbias = CCDData.read('RBIAS.fits', ignore_missing_end=True)
      gc.collect()

   if not(rflat_list==[]):
      master_redflat = create_masterflat(rflat_list)#if bias was created it should probably be "create_masterflat(rflat_list,masterbias=master_rluebias)", but it generates an error
      master_redflat.write('RFLAT.fits', clobber=True)
      del master_redflat
      gc.collect()
      print("\nCreated red masterflat\n")

   if not(rarc_list==[]): 
      master_redarc = create_masterflat(rarc_list)
      master_redarc.write('RARC.fits', clobber=True)
      del master_redarc
      gc.collect()
      print("\nCreated red masterarc\n")
   del master_redbias 
   gc.collect()

   return True








def write_orderframe(f_limit_red=1000.0,f_limit_blue=1000.0,interactive=True):
   """Creates blue and red order frames

   Parameters
   ----------
   f_limit_red: float
       Limiting value for detecting orders in the red flat. Value should be higher than counts of the background light, but lower than in most part of any order.
   f_limit_blue: float
       Same as f_limit_red, but for blue frame.
   interactive: bool
       If true the program will display the flat frames in DS9 and it will ask the user to enter values of f_limit_red and f_limit_blue 

   Returns
   -------
      Writes HNORM.fits RNORM.fits HORDER.fits and RORDER.fits in the current directory

   """
   print("Creating normalized red flat")
   if interactive:

      import pyds9
      
      d=pyds9.DS9()
      d.set("file RFLAT.fits")
      print("Flat opened in DS9")

      check=True
      while check:
         f_limit_red=raw_input("Enter a count number that is higher than background light, but lower than in most part of any order: ")
         try:
            f_limit_red=float(f_limit_red)
            check=False
         except ValueError:
            print("\nError: Entered value should be a float ")
         
   master_redflat = CCDData.read('RFLAT.fits', ignore_missing_end=True)
   image = nd.filters.maximum_filter(master_redflat, 10)
   mask = (image > f_limit_red)
   norm = normalize_image(image, mod.models.Legendre1D(10), mask=mask)
   norm[norm < f_limit_red]=0
   hdu = fits.PrimaryHDU(norm)
   hdu.writeto('RNORM.fits', clobber=True)

   #create the initial detection kernal
   ys, xs = norm.shape
   xc = int(xs/2.0)
   norm[norm>500] = 500
   ndata = norm[:,xc]
   detect_kern = ndata[5:80]
   #these remove light that has bleed at the edges and may need adjusting
   norm[:,:20]=0
   norm[:,4040:]=0

   #detect orders in image

   frame = create_orderframe(norm, 53, xc, detect_kern, y_start=8,
                             y_limit=3920, smooth_length=20)

   hdu = fits.PrimaryHDU(frame)
   hdu.writeto('RORDER.fits', clobber=True)

      
   print("Creating normalized blue flat")
   if interactive:
      d.set("file HFLAT.fits")
      print("Flat opened in DS9")
      
      check=True
      while check:
         f_limit_blue=raw_input("Enter a count number that is higher than background light, but lower than in most part of any order: ")
         try:
            f_limit_blue=float(f_limit_blue)
            check=False
         except ValueError:
            print("\nError: Entered value should be a float ")

   #normalize the blue image--these values may need adjusting based on the image
   master_blueflat = CCDData.read('HFLAT.fits', ignore_missing_end=True)
   image = nd.filters.maximum_filter(master_blueflat, 10)
   mask = (image > f_limit_blue)
   norm = normalize_image(image, mod.models.Legendre1D(10), mask=mask)
   norm[norm < f_limit_blue]=0
   hdu = fits.PrimaryHDU(norm)
   hdu.writeto('HNORM.fits', clobber=True)



   #create the initial detection kernal
   ys, xs = norm.shape
   xc = int(xs/2.0)
   norm[norm>500] = 500
   ndata = norm[:,xc]
   detect_kern = ndata[5:80]
   #these remove light that has bleed at the edges and may need adjusting
   norm[:,:20]=0
   norm[:,4040:]=0

   #detect orders in image

   frame = create_orderframe(norm, 53, xc, detect_kern, y_start=8,
                             y_limit=3920, smooth_length=20)

   hdu = fits.PrimaryHDU(frame)
   hdu.writeto('HORDER.fits', clobber=True)

   return True





if __name__=='__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('-l','--list', help='Text file containing list of raw calibration files',type=str, required=False)
   parser.add_argument("-i","--interactive",help="Interactive fitting",action="store_true")
   parser.add_argument("-o","--orderframe",help="The program will create the order frame",action="store_true")
   parser.add_argument("-p","--parameters", nargs='+',help="Parameters for making the order frame: f_limit_red and f_limit_blue", required=False)
   args=parser.parse_args()

   check_input=True
   if (args.orderframe) and (not args.interactive) and (not len(args.parameters)==2):
      check_input=False
      print("In order to create the order frame the script should be run in interactive mode or parameters for making order frame should be specified")
    
   if (not args.list==None):
      blue_files,red_files=divide_list(args.list)
   else:
      blue_files,red_files=create_list_folder()

   if blue_files+red_files==[]:
      check_input=False
      if (not args.list==None):
         print("No HRS images in the list")
      else:
         print("No HRS images found in the running directory")

   if check_input==True:
      write_bias_flat_arc(blue_files+red_files)
      if args.orderframe and args.interactive:
         write_orderframe(interactive=True)
      elif args.orderframe and len(args.parameters)==2:
         write_orderframe(f_limit_red=args.parameters[0],f_limit_blue=args.parameters[1])
