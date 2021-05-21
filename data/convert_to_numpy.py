import xarray as xr
import numpy as np

ds = xr.load_dataset("single_pos.nc")
im = ds.images.sel(C='SRS')
masks = ds.mask.sel(T=0).data[0]

im = ds.images.sel(T=0, C='SRS', R=0).data[0]
mask = ds.mask.sel(T=0).data[0]


np.save("srs_image_1.npy", im)
np.save("mask_image_1.npy", mask)
