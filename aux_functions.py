#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:13:13 2021

@author: michael
"""
# Python enviroment imports
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime
import glob
import numpy as np
import pandas as pd
import shutil
# Local imports

# File functions
    
def clean_temp_files():
    """
    Removes all files temporary files (i.e. that end with _temp) in the working
    directory
    """
    temp_files = glob.glob('*_temp')
    confirm = 0
    confirm = input(
        'Confirm removing temporary files. Yes=1; No=Any other integer (You will need to remove them by hand to rerun '
        'the script): ')
    if confirm == 1:
        for temp_file in temp_files:
            shutil.rmtree(temp_file, ignore_errors=True)
        print(str('Removing ' + str(len(temp_files)) + ' temporary files.'))
        return 0
    else:
        print('No files removed')
        return 0


def load_fits(fits_path, header_ext=0, return_path=False, fix_wcs=True):
    file = fits_path.split('/')[-1]
    print('Loaded file: ', file)
    print('Header Extension: ', header_ext)
    fits_file = fits.open(fits_path)
    data = fits_file[header_ext].data
    print('Image size: ', data.shape)
    header = fits_file[header_ext].header
    if fix_wcs:
        w = WCS(header).to_header()
        header.update(w)
    fits_file.close()
    if return_path:
        print('Path returned: ', fits_path)
        return data, header, fits_path
    else:
        return data, header

def read_setup_file(setup_path):
    """
    Reads the setup and parses all required arguments to the following functions
    """
    df_params = pd.read_csv(setup_path, header=0)
    print('Loaded Parameter File:')
    print(df_params)
    return df_params

# Header functions

def edit_changes_header(changes_head):
    for i in [3, 4]:
        for key in ['NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
            changes_head.remove('{}{}'.format(key, i), ignore_missing=True,
                                remove_all=True)
            changes_head.remove('LTYPE', ignore_missing=True)
            changes_head.remove('LSTART', ignore_missing=True)
            changes_head.remove('LSTEP', ignore_missing=True)
            changes_head.remove('LWIDTH', ignore_missing=True)
            changes_head.remove('LONPOLE', ignore_missing=True)
            changes_head.remove('LATPOLE', ignore_missing=True)
            changes_head.remove('RESTFRQ', ignore_missing=True)
            changes_head.remove('WCSAXES', ignore_missing=True)
    changes_head.remove('HISTORY', remove_all=True)
    changes_head.remove('SPECSYS', ignore_missing=True)
    changes_head.remove('ALTRVAL', ignore_missing=True)
    changes_head.remove('ALTRPIX', ignore_missing=True)
    changes_head.remove('VELREF', ignore_missing=True)
    changes_head.remove('EXTEND', ignore_missing=True)
    changes_head.remove('PC004001', ignore_missing=True)
    changes_head.remove('PC004002', ignore_missing=True)
    changes_head.remove('PC004003', ignore_missing=True)
    changes_head.remove('PC004004', ignore_missing=True)
    changes_head.remove('PC002001', ignore_missing=True)
    changes_head.remove('PC001002', ignore_missing=True)
    changes_head.remove('PC003002', ignore_missing=True)
    changes_head.remove('PC001004', ignore_missing=True)
    changes_head.remove('PC003004', ignore_missing=True)
    changes_head.remove('PC003001', ignore_missing=True)
    changes_head.remove('PC001003', ignore_missing=True)
    changes_head.remove('PC003003', ignore_missing=True)
    changes_head.remove('PC002002', ignore_missing=True)
    changes_head.remove('PC002004', ignore_missing=True)
    changes_head.remove('PC002003', ignore_missing=True)
    changes_head.remove('PC4_1', ignore_missing=True)
    changes_head.remove('PC4_2', ignore_missing=True)
    changes_head.remove('PC2_3', ignore_missing=True)
    changes_head.remove('PC4_3', ignore_missing=True)
    changes_head.remove('PC2_4', ignore_missing=True)
    changes_head.remove('PC4_4', ignore_missing=True)
    changes_head.remove('PC2_1', ignore_missing=True)
    changes_head.remove('PC1_2', ignore_missing=True)
    changes_head.remove('PC3_2', ignore_missing=True)
    changes_head.remove('PC3_3', ignore_missing=True)
    changes_head.remove('PC3_4', ignore_missing=True)
    changes_head.remove('PC3_1', ignore_missing=True)
    changes_head.remove('PC1_3', ignore_missing=True)
    changes_head.remove('PV2_1', ignore_missing=True)
    changes_head.remove('PV2_2', ignore_missing=True)
    changes_head.remove('PC02_01', ignore_missing=True)
    changes_head.remove('PC04_01', ignore_missing=True)
    changes_head.remove('PC02_02', ignore_missing=True)
    changes_head.remove('PC04_02', ignore_missing=True)
    changes_head.remove('PC02_03', ignore_missing=True)
    changes_head.remove('PC04_03', ignore_missing=True)
    changes_head.remove('PC02_04', ignore_missing=True)
    changes_head.remove('PC04_04', ignore_missing=True)
    changes_head.remove('PC03_01', ignore_missing=True)
    changes_head.remove('PC03_02', ignore_missing=True)
    changes_head.remove('PC03_03', ignore_missing=True)
    changes_head.remove('PC03_04', ignore_missing=True)
    changes_head.remove('PC01_02', ignore_missing=True)
    changes_head.remove('PC01_04', ignore_missing=True)
    changes_head.remove('PC01_03', ignore_missing=True)

    changes_head['NAXIS'] = 2
    for key in changes_head.keys():
        if len(key) > 0:
            if key[0:2] == 'PC':
                changes_head.remove(key)
            if key[0:3] == 'OBS':
                changes_head.remove(key, ignore_missing=True)
            if key[2:4] == '00':
                changes_head.remove(key, ignore_missing=True)

    changes_head['NAXIS'] = 2
    return changes_head


def date_to_header(header):
    """
    Adds a comment line to the fits header which states the date and origin 
    of the file

    Parameters
    ----------
    header : fits header

    Returns
    -------
    None.

    """
    date = datetime.now().strftime('%m/%d/%y')
    header['COMMENT'] = 'File written on ' + date + ' by M. Stein (AIRUB)'




def clip_fits(infile, outfile, clip_val, hdu=0):
    data, header = load_fits(infile, header_ext=hdu)
    header['COMMENT'] = 'Clipped at: ' + str(clip_val)
    data_clip = clip_value(data=data, value=clip_val)
    date_to_header(header)
    fits.writeto(outfile, data=data_clip, header=header, overwrite=True)
    return

# Array funtions

def clip_value(data, value):
    data_clipped = data[np.where(data <= value)] = np.nan
    return data_clipped

def fill_nans(infile, outfile, fill_value=0.0, hdu=0):
    fits_file = fits.open(infile)
    data = fits_file[hdu].data
    header = fits_file[hdu].header
    data_filled = np.nan_to_num(data, nan=fill_value)
    fits.writeto(outfile, data=data_filled, header=header)

def masked_array_to_filled_nan(masked_array):
    if isinstance(masked_array, np.ma.MaskedArray):
        filled_array = masked_array.filled(fill_value=np.nan)
    else:
        filled_array = masked_array
    return filled_array