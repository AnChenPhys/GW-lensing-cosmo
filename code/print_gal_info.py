import fitsio
import pandas as pd

cat_fname = '../data/22137.fits'
npcat = fitsio.read(cat_fname)#, columns=columns)
npcat = npcat.byteswap().newbyteorder()
data_frame = pd.DataFrame.from_records(npcat)

print(data_frame)