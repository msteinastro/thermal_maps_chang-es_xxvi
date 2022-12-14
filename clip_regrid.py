from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from reproject import reproject_exact
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np


# Clipping & Regridding
def clipping(data, nsigma,  show_plots=False):
    mean, median, std = sigma_clipped_stats(data=data, sigma=3)
    print('Data mean, std: ', mean, std)
    clipped_data = np.ma.masked_where(data < mean+nsigma*std, data)
    # Astropy cannot write masked arrays yet. Convert to normal array with nan's.
    if show_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(data)
        ax1.set_title('Data')
        ax2.imshow(clipped_data)
        ax2.set_title('Clipped Data')
        plt.show()
    return clipped_data

def clip_value(data, value):
    data_clipped = data[np.where(data <= value)] = np.nan
    return data_clipped

def regrid_halpha(halpha_data, halpha_header, wise_header):
    halpha_regrid = reproject_exact(input_data=(halpha_data, halpha_header), output_projection=wise_header,
                                    return_footprint=False, parallel=True)
    w = WCS(wise_header).to_header()
    halpha_header.update(w)
    return halpha_regrid, halpha_header


def regrid(data, header, target_header, flux_conserve=True, keep_old_header=False, print_head=False):
    data_regrid = reproject_exact(input_data=(data, header), output_projection=target_header,
                                    return_footprint=False, parallel=True)
    w = WCS(target_header).to_header()
    if keep_old_header:
        new_header = header.copy()
        new_header.update(w)
    else:
        new_header = fits.PrimaryHDU().header
        new_header.update(w)
        new_header['NAXIS'] = 2
        new_header.insert('NAXIS', ('NAXIS1', target_header['NAXIS1']), after=True)
        new_header.insert('NAXIS1', ('NAXIS2', target_header['NAXIS2']), after=True)
    if print_head:
        print(repr(new_header))
    if flux_conserve:
        sum_before = np.nansum(data)
        sum_regrid = np.nansum(data_regrid)
        print('Sum of data before regridding:', sum_before)
        print('Sum of data after regridding:', sum_regrid)
        print('Multiply data with to conserve total flux:', sum_before / sum_regrid)
        data_regrid = data_regrid * (sum_before / sum_regrid)
    return data_regrid, new_header