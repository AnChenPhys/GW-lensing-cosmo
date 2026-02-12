import numpy as np
import matplotlib.pyplot as plt
import fitsio
import pandas as pd


columns =  ['ra_gal','dec_gal','lml_r','lmstellar','gr_gal','z_cgal_v','cgal_v','sdss_r_abs_mag','sdss_r_true']

cat_fname = '../data/micecat.fits'
npcat = fitsio.read(cat_fname, columns=columns)
npcat = npcat.byteswap().newbyteorder()
data_frame = pd.DataFrame.from_records(npcat)

ra_gal = np.array(data_frame['ra_gal'])
dec_gal = np.array(data_frame['dec_gal'])
z_gal = np.array(data_frame['z_cgal_v'])
m_gal = np.array(data_frame['sdss_g_true'])

ra = 26
dec = 48
delta = 1.7

index_ra = np.where(ra_gal[np.where(ra_gal>ra-delta)[0]]<ra+delta)[0]
index_ra_dec = np.where(dec_gal[index_ra][np.where(dec_gal[index_ra]>dec-delta)[0]]<dec+delta)[0]

z_ind = z_gal[index_ra_dec]
m_ind = m_gal[index_ra_dec]
lml_ind = np.array(data_frame['lml_r'])[index_ra_dec]
lm_ind = np.array(data_frame['lmstellar'])[index_ra_dec]

np.savetxt('zgal.txt',z_ind)

plt.figure()
plt.hist(z_ind, bins=30)
plt.savefig('../plots/z_gal.png')

index_z = np.where(z_ind<0.1)[0]

plt.figure()
plt.hist(z_ind[index_z], bins=30)
plt.savefig('../plots/z_gal_in.png')

np.savetxt('zgal_mgal_in.txt',np.array([z_ind[index_z],m_ind[index_z],lml_ind[index_z],lm_ind[index_z]]).T)

