
import sys
from astropy.io import fits
from astropy.stats import sigma_clip

import pylab as pl
import numpy as np

def write_spdict(outfile, sp_dict):

    o_arr = None
    w_arr = None
    f_arr = None

    for k in sp_dict.keys():
        w,f = sp_dict[k]
        if w_arr is None:
            w_arr = 1.0*w
            f_arr = 1.0*f
            o_arr = k*np.ones_like(w, dtype=int)
        else:
            w_arr = np.concatenate((w_arr, w))
            f_arr = np.concatenate((f_arr, f))
            o_arr = np.concatenate((o_arr, k*np.ones_like(w, dtype=int)))

    c1 = fits.Column(name='Wavelength', format='D', array=w_arr, unit='Angstroms')
    c2 = fits.Column(name='Flux', format='D', array=f_arr, unit='Counts')
    c3 = fits.Column(name='Order', format='I', array=o_arr)

    tbhdu = fits.BinTableHDU.from_columns([c1,c2,c3])
    tbhdu.writeto(outfile, clobber=True)


outfile = sys.argv[1]
inlist = sys.argv[2:]

hdu_list = [fits.open(x) for x in inlist]

min_order = hdu_list[0][1].data['ORDER'].min()
max_order = hdu_list[0][1].data['ORDER'].max()

sp_dict={}
for order in range(min_order, max_order):
    data_list = []
    mdata_list = []
    mean = None
    for hdu in hdu_list:
        w = hdu[1].data['Wavelength']
        f = hdu[1].data['Flux']
        o = hdu[1].data['Order']
        wave = w[o==order]
        data = f[o==order]
        if mean==None: mean = data.mean()
        data_list.append(data)
        data = data * mean / data.mean()
        mdata_list.append(data)
    data = np.array(data_list)
    mdata = np.array(mdata_list)
    mdata_ave = mdata.mean(axis=0)
    flux = np.zeros(len(mdata_ave))
    count = np.zeros(len(mdata_ave))
    for i in range(len(flux)):
        s=mdata_ave[i]
        e=mdata_ave[i]**0.5/len(mdata_list)
        for j in range(len(data)):
            if mdata[j][i] < s+1.5*e:
               flux[i] += mdata[j][i]
               count[i] += 1
    count[count==0] = 1
    flux = flux / count
    sp_dict[order] = [wave, flux]
    
write_spdict(outfile, sp_dict)
