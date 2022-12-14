#!/usr/bin/env python3
# -*- coding: utf-8 -*-
WISE_comment = 'Data convolved to 20 arcsec Gauss PSF using a Pypher Kernel'

# Python enviroment imports
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
from astropy import wcs
from radio_beam import Beam
from astropy.nddata import block_reduce
import numpy as np
import os
import astropy.units as u 
import datetime
from aux_functions import date_to_header
from math_physics import fwhm_to_stddev

def convolve_data(data, header, conv_kernel, write_files=False, fits_name=None, fits_name_end=None,
                  header_comment=WISE_comment, subfolder='wise_convolved', beam_factor=1.):
    print('Start convolution')
    print('Shape of Convolution Kernel: ', conv_kernel.shape)
    data_conv = convolve_fft(data, conv_kernel, normalize_kernel=True, preserve_nan=True, fill_value=np.nan,
                             allow_huge=True)
    print('Convolution completed')
    # Data needs to be corrected by area_beam_1/area_beam_2 if data is in Jy/beam
    if beam_factor != 1.:
        print('Multiply data with %.2f to account for Beamsize' % beam_factor)
        print('Array Mean:', np.nanmean(data_conv))
        data_conv = data_conv * beam_factor
        print('Array Mean after Beam Correction:', np.nanmean(data_conv))

    if write_files:
        file_name = fits_name.split('/')[-1]
        file_clean = file_name.split('.')[0]
        header['COMMENT'] = header_comment
        date = datetime.now().strftime('%m/%d/%y')
        header['COMMENT'] = 'File written on ' + date + ' by M. Stein (AIRUB)'
        if not os.path.exists(out_dir + '/' + subfolder):
            os.mkdir(out_dir + '/' + subfolder)
        out_str = out_dir + '/' + subfolder + '/' + file_clean + fits_name_end
        fits.writeto(filename=out_str, data=data_conv, header=header, overwrite=True)

    return data_conv

def convolve_to_beam(target_res_arcsec, data, header,
                     fits_name=None, fits_name_end=None,
                     header_comment=None, subfolder='dummy'
                     ):
    head_wcs = WCS(header)
    pix_scale = wcs.utils.proj_plane_pixel_scales(head_wcs)
    print(pix_scale)
    if pix_scale[0] != pix_scale[1]:
        print('Pixel Dimensions are not the same:', pix_scale)
    tar_beam = Beam(target_res_arcsec * u.arcsec)
    cur_beam = Beam.from_fits_header(header)
    print('Major Axis Current Beam', cur_beam.major.to(u.arcsec))
    print('Major Axis Target Beam', tar_beam.major)
    if cur_beam.major.to(u.arcsec) > tar_beam.major:
        print('WARNING: Current beam is larger than target beam. Adapt target beam.')
        tar_beam = Beam(cur_beam.major.to(u.arcsec))
    deconv_beam = tar_beam.deconvolve(cur_beam)
    print('Current Beam:', cur_beam)
    print('Target Beam:', tar_beam)
    print('Deconvolved Beam:', deconv_beam)
    # Correction Factor needed if Map is in Jy/Beam
    beam_axis_prod_tar = tar_beam.major.to(u.arcsec) * tar_beam.minor.to(u.arcsec)
    beam_axis_prod_cur = cur_beam.major.to(u.arcsec) * cur_beam.minor.to(u.arcsec)
    beam_factor = (beam_axis_prod_tar / beam_axis_prod_cur)
    print('Beam Factor:', beam_factor.value)
    beam_factor = beam_factor.value
    kernel_scale= pix_scale[0] * u.degree
    deconv_kernel = deconv_beam.as_kernel(pixscale=kernel_scale)
    date_to_header(header)
    header = tar_beam.attach_to_header(header)
    convolve_data(data=data, header=header, conv_kernel=deconv_kernel, beam_factor=beam_factor, write_files=True,
                  header_comment=header_comment, fits_name=fits_name, fits_name_end=fits_name_end, subfolder=subfolder)

def down_sample_kernel(psf_kernel, wise_compare, manual=False, down_block=(0, 0)):
    """
    :param psf_kernel:
    :param wise_compare:
    :param manual:
    :param down_block:
    :return:
    """
    print('Downscale the PSF kernel to match the WISE data')
    print('Try to load Kernel from:', psf_kernel)
    kernel = fits.open(psf_kernel)
    kernel_dat = kernel[0].data
    kernel_head = kernel[0].header
    w_kernel = WCS(kernel_head)
    ker_pix_x, ker_pix_y = w_kernel.wcs.cd[0, 0], w_kernel.wcs.cd[1, 1]
    if manual:
        print('Block size: ', down_block)
        kernel_dat_down_sample = block_reduce(data=kernel_dat, block_size=down_block)
    else:
        wise_map = fits.open(wise_compare)
        wise_head = wise_map[0].header
        w_wise = WCS(wise_head)
        wise_pix_x, wise_pix_y = w_wise.wcs.cdelt[0], w_wise.wcs.cdelt[1]
        combine_x = int(np.abs(np.round(wise_pix_x / ker_pix_x)))
        combine_y = int(np.abs(np.round(wise_pix_y / ker_pix_y)))
        print('Block size: ', (combine_x, combine_y))
        kernel_dat_down_sample = block_reduce(data=kernel_dat, block_size=(combine_x, combine_y))
    print('New PSF shape: ', kernel_dat_down_sample.shape)
    print('Sum after downscaling: ', np.sum(kernel_dat_down_sample))
    return kernel_dat_down_sample

def make_halpha_kernel(header, fwhm_arcsec):
    w_halpha = WCS(header)
    scale = wcs.utils.proj_plane_pixel_scales(w_halpha)
    mean_scale = np.mean(scale) * u.degree
    scale_arcsec = mean_scale.to(u.arcsec)
    fwhm_pixel = fwhm_arcsec/scale_arcsec.value
    std_pixel = fwhm_to_stddev(fwhm_pixel)
    kernel = Gaussian2DKernel(std_pixel)
    return kernel
