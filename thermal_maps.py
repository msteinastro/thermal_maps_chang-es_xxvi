#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# imports from other repositories
import __init__
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.stats import sigma_clipped_stats
from astropy import units as u
import astropy.wcs as wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from datetime import datetime
import glob
import os
from scipy import constants
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from radio_beam import Beam
from radio_beam.utils import deconvolve
from regions import read_ds9
from reproject import reproject_exact
# internal imports
from aux_functions import read_setup_file, load_fits, masked_array_to_filled_nan
from calibration import wise_band4_cal, halpha_flux_cal
from clip_regrid import regrid, clipping
from convolution import convolve_data, down_sample_kernel, make_halpha_kernel
from math_physics import parsec_to_cm, jy_to_erg_s_cm_cm, wavelength_to_freq,\
     sfr_murphy_2011, thermal_murphy_2011, erg_s_cm_cm_hz_to_jy, jy_to_jy_beam, jy_beam_to_jy, fwhm_to_stddev
import make_plots

# Used constants:
temp_dir = 'tmp/'
out_dir = 'output/'
plot_dir = 'output/plots/'
WISE_comment = 'Data convolved to 20 arcsec Gauss PSF using a Pypher Kernel'
# IR weighting for Halpha correction:
a_new = 0.042
nu_wise4 = wavelength_to_freq(22e-6)  # 22 micron data (w4 band)
nu_24mu = wavelength_to_freq(24e-6)
nuGHzC = 5.99  # freq of C band in GHz
nuGHzL = 1.5   # L band
nuMHzLoTTS = 144  # MHz


def make_thermal(setup_file, do_clip=False, nsig_alpha=3, nsig_wise=3,
                 do_mask=True, do_cal=True, do_convolution=True, do_regrid_halpha=True,
                 do_regird_to_changes=True, do_regrid_to_main=True, reduce_size=False, target_res=15,
                 write_interim_files=False, write_non_thermal=True,
                 show_plots=False, halpha_in_erg_s=False, with_lofar=False, run_tag=None,
                 out_dir=out_dir, plot_dir= plot_dir,
                 Pypher_kernel='/home/michael/PycharmProjects/thermal_maps/outs/w4cut_to_15arcsec.fits',
                 WISE_compare='/data/phd_data/aux_maps/WISE-WERGA/skysubtracted/22micron/ngc2683.w4.ss.fits'):
    for folder in [temp_dir, out_dir, plot_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    if run_tag is not None:
        out_dir = out_dir + run_tag
        plot_dir = plot_dir + run_tag
        print(f"New out_dir: {out_dir}")
        print(f"New plot_dir: {plot_dir}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            os.makedirs(out_dir + '/thermal_maps')
            os.makedirs(out_dir + '/converted_radio')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)


        #print(out_dir)
            #exit()
    parameters = read_setup_file(setup_file)
    # Load wise pypher kernel
    wise_kernel = down_sample_kernel(psf_kernel=Pypher_kernel, wise_compare=WISE_compare)
    for index, row in parameters.iterrows():
        # Parse setup:
        galaxy = row['galaxy']
        print('-----------------Working on Galaxy: ', galaxy)
        distance = row['distance[Mpc]']
        coord = row['coord']
        size = row['size[arcmin]']
        halpha = row['halpha_path']
        wise = row['wise_path']
        halpha_flux_fac = row['halpha_flux_cal']
        mask = row['mask_path']
        radio_maps = row['radio_maps'].strip("[]").split(",")
        radio_maps_ident = row['radio_maps_ident'].strip("[]").split(",")
        therm_freq = row['list_therm'].strip("[]").split(",")
        therm_freq_ident = row['list_therm_ident'].strip("[]").split(",")
        # Prior Checks:
        n_radio = len(radio_maps)
        n_radio_ident = len(radio_maps_ident)
        if n_radio < 1:
            raise ValueError('You need to pass at least one radio map')
        if n_radio != len(radio_maps_ident):
            raise ValueError(f'You need to pass as many radio maps ({n_radio} passed) as radio map identifiers ({n_radio_ident} passed)')
        n_therm = len(therm_freq)
        if n_therm < 1:
            raise ValueError('You must specify at least one frequency for which a thermal emission map is to be calculated.')
        if n_therm != len(therm_freq_ident):
            raise ValueError('You need to pass as many frequencies to compute thermal maps as thermal frequency identifiers')
        # Load fits files
        halpha_dat, halpha_head = load_fits(fits_path=halpha, fix_wcs=True)
        w_halpha = WCS(halpha_head)
        wise_dat, wise_head = load_fits(fits_path=wise, fix_wcs=True)
        w_wise = WCS(wise_head)
        print(f'Load the main radio map: {radio_maps[0]}')
        main_radio_dat, main_radio_head = load_fits(fits_path=radio_maps[0])
        main_radio_filename = radio_maps[0].split('/')[-1]
        main_radio_filename_clean = main_radio_filename.split('.')[0]
        
        print('Loaded  all data')
        print('Max Halpha', np.nanmax(halpha_dat))
        print('Max WISE', np.nanmax(wise_dat))
       
        # Compute distance sphere to convert from surface brightness to luminosities
        d_cm = parsec_to_cm(distance * 1e6)
        luminosity_sphere = 4 * np.pi * d_cm ** 2
        print('Distance Sphere[cm^2]: ', luminosity_sphere)
        if do_clip:
            halpha_dat = clipping(data=halpha_dat, nsigma=nsig_alpha, show_plots=show_plots)
            wise_dat = clipping(data=wise_dat, nsigma=nsig_wise, show_plots=show_plots)
        if do_mask:
            # Load Mask to mask WISE and Halpha data
            ds9_reg = read_ds9(mask)
            ds9_reg_pix_wise = ds9_reg[0].to_pixel(w_wise)
            ds9_reg_pix_halpha = ds9_reg[0].to_pixel(w_halpha)
            wise_mask = ds9_reg_pix_wise.to_mask().to_image(wise_dat.shape)
            wise_mask_sky = np.invert(wise_mask.astype('bool'))
            halpha_mask = ds9_reg_pix_halpha.to_mask().to_image(halpha_dat.shape)
            halpha_mask_sky = np.invert(halpha_mask.astype('bool'))
            wise_dat = np.ma.masked_array(data=wise_dat, mask=wise_mask_sky)
            halpha_dat = np.ma.masked_array(data=halpha_dat, mask=halpha_mask_sky)
        if do_cal:
            # Flux calibration
            # IR (WISE -> 24mu)
            wise_dat_cal = wise_band4_cal(data=wise_dat, star_forming=True)
            # Convert to 24mu using empirical relation 24mu = 1.03*22mu (Wiegert et. al.)
            ir_24mu_jy = wise_dat_cal * 1.03
            #  Convert 24mu data from Jy to erg/s/cm^2
            ir_24_mu_erg_s_cm_cm = jy_to_erg_s_cm_cm(ir_24mu_jy, frequency=nu_24mu)
            # H alpha
            halpha_dat = halpha_flux_cal(data=halpha_dat, factor=halpha_flux_fac)
        else:
            print('Input 24mu data should be in Jansky per pix!')
            print('Input Halpha data should be in erg/(s*cm^2) per pix')
            ir_24_mu_erg_s_cm_cm = jy_to_erg_s_cm_cm(wise_dat, frequency=nu_24mu)
        print('After Calibration (Data unit: erg/(s*cm^2) per pix)')
        print('Brightest Pixel H alpha', np.nanmax(halpha_dat))
        print('Brightest Pixel 24mu', np.nanmax(ir_24_mu_erg_s_cm_cm))

        if write_interim_files:
            ir_24_mu_erg_s_cm_cm_filled = masked_array_to_filled_nan(ir_24_mu_erg_s_cm_cm)
            halpha_dat_filled = masked_array_to_filled_nan(halpha_dat)
            fits.writeto(filename=out_dir + '/ir24mu_after_cal.fits', data=ir_24_mu_erg_s_cm_cm_filled,
                         overwrite=True)
            fits.writeto(filename=out_dir + '/halpha_after_cal.fits', data=halpha_dat_filled, overwrite=True)
        # Fill masked arrays with nan's for astropy convolution
        halpha_dat = masked_array_to_filled_nan(halpha_dat)
        ir_24_mu_erg_s_cm_cm = masked_array_to_filled_nan(ir_24_mu_erg_s_cm_cm)
        if do_convolution:
            print('Convolve the data to common Beam:')
            print('Halpha & IR sum before convolution: ', np.nansum(halpha_dat), np.nansum(ir_24_mu_erg_s_cm_cm))
            halpha_kernel = make_halpha_kernel(header=halpha_head, fwhm_arcsec=target_res)
            halpha_dat = convolve_data(data=halpha_dat, header=halpha_head, conv_kernel=halpha_kernel)
            ir_24_mu_erg_s_cm_cm = convolve_data(data=ir_24_mu_erg_s_cm_cm, header=wise_head, conv_kernel=wise_kernel)
            print('Halpha & IR sum after convolution: ', np.nansum(halpha_dat), np.nansum(ir_24_mu_erg_s_cm_cm))
            if write_interim_files:
                ir_24_mu_erg_s_cm_cm_filled = masked_array_to_filled_nan(ir_24_mu_erg_s_cm_cm)
                halpha_dat_filled = masked_array_to_filled_nan(halpha_dat)
                fits.writeto(filename=out_dir + '/ir24mu_after_conv.fits', data=ir_24_mu_erg_s_cm_cm_filled,
                             overwrite=True)
                fits.writeto(filename=out_dir + '/halpha_after_conv.fits', data=halpha_dat_filled, overwrite=True)

        #  Regrid H alpha data to WISE frame
        if do_regrid_halpha:
            halpha_dat_regrid_flux_conserve, halpha_head_regrid = regrid(data=halpha_dat, header=halpha_head,
                                                                     target_header=wise_head, flux_conserve=True)
        else: halpha_dat_regrid_flux_conserve, halpha_head_regrid = halpha_dat, halpha_head
        #  Edit headers:
        wise_head_erg = wise_head.copy()
        wise_head_erg['BUNIT'] = 'erg/s/cm/cm'
        beam = Beam(15*u.arcsec)
        halpha_head_regrid = beam.attach_to_header(halpha_head_regrid)

        if write_interim_files:
            out_string_halpha = str(out_dir + '/' + galaxy + '_halpha_cal.fits')
            halpha_filled = masked_array_to_filled_nan(halpha_dat_regrid_flux_conserve)
            fits.writeto(filename=out_string_halpha, data=halpha_filled, header=halpha_head_regrid,
                         overwrite=True)
            out_string_wise = str(out_dir + '/' + galaxy + '_wise_cal.fits')
            wise_filled = masked_array_to_filled_nan(wise_dat)
            fits.writeto(filename=out_string_wise, data=wise_filled, header=wise_head_erg,
                         overwrite=True)


        # Convert maps to luminosities
        ir_24_mu_erg_s = ir_24_mu_erg_s_cm_cm * luminosity_sphere
        halpha_erg_s = halpha_dat_regrid_flux_conserve * luminosity_sphere
        if halpha_in_erg_s:
            halpha_erg_s = halpha_dat_regrid_flux_conserve
        print('Brightest Pixel 24mu', np.nanmax(ir_24_mu_erg_s))
        if write_interim_files:
            ir_24_mu_erg_s_filled = masked_array_to_filled_nan(ir_24_mu_erg_s)
            halpha_erg_s_filled = masked_array_to_filled_nan(halpha_erg_s)
            fits.writeto(filename=out_dir+'/ir24mu_erg_s.fits', data=ir_24_mu_erg_s_filled, overwrite=True)
            fits.writeto(filename=out_dir+'/halpha_erg_s.fits', data=halpha_erg_s_filled, overwrite=True)

        print('Brightest Pixel H alpha', np.nanmax(halpha_erg_s))
        halpha_corr_mix = halpha_erg_s+(a_new*ir_24_mu_erg_s)
        print('Halpha corrected')
        print('Max Halpha_corr', np.nanmax(halpha_corr_mix))
        if write_interim_files:
            correction_diff = (halpha_corr_mix - halpha_erg_s)/halpha_corr_mix
            fits.writeto(filename=out_dir+'/rel_halpha_corr_halpha_diff.fits', data=correction_diff, overwrite=True)
        sfr_map = sfr_murphy_2011(halpha_corr_mix)
        sfr_head = wise_head.copy()
        sfr_head['BUNIT'] = 'M_sol/year'
        fits.writeto(filename=out_dir+'/'+galaxy+'sfr_map.fits',
                     data=sfr_map, header=sfr_head,overwrite=True)
        for index, freq in enumerate(therm_freq):
            freq = float(freq)
            print(f"Creating a thermal map for the frequency: {freq} GHz.")
            therm_map = thermal_murphy_2011(t_electron=1e4, freq_ghz=freq, sfr=sfr_map)
            therm_map_jy = erg_s_cm_cm_hz_to_jy(therm_map, distance_sphere=luminosity_sphere)
            therm_head_jy = halpha_head_regrid.copy()
            date = datetime.now().strftime('%m/%d/%y')
            therm_head_jy['BUNIT'] = 'Jy'
            therm_head_jy['COMMENT'] = 'File written on ' + date + ' by M. Stein (AIRUB)'
            therm_head_jy['COMMENT'] = 'Thermal Emission computed for ' + str(therm_freq_ident[index]) + 'GHz.'
            if write_interim_files:
                fits.writeto(filename=out_dir+'/thermal_maps/'+galaxy+'_thermal_' + therm_freq_ident[index] + '.fits',
                         data=therm_map_jy, header=therm_head_jy, overwrite=True, output_verify='silentfix')
            if do_regrid_to_main:
                print('Reproject thermal map to the main radio map')
                thermal_jy_repr, head_thermal_jy_repr = regrid(data=therm_map_jy, header=halpha_head_regrid,
                                                   target_header=main_radio_head, keep_old_header=True)
                fits.writeto(filename=out_dir+'/thermal_maps/'+galaxy+'_thermal_regrid_'+ therm_freq_ident[index] + '.fits',
                            data=thermal_jy_repr, header=head_thermal_jy_repr, overwrite=True, output_verify='silentfix')
        # Regrid the radio data to the main radio map, if multiple radio maps were given:
        if n_radio > 1:
            # Skip first map as it is the main map.
            for index, radio_map in enumerate(radio_maps[1:], start=1):
                print(f'Load file: {radio_map}')
                radio_dat, radio_head = load_fits(fits_path=radio_map)
                radio_filename = radio_map.split('/')[-1]
                radio_filename_clean = radio_filename.split('.')[0]
                print('Regrid additional map to  the main radio frame')
                # Jy/Beam map: do not conserve sum of pixel values!
                radio_regrid_dat, radio_regrid_head = regrid(data=radio_dat, header=radio_head, target_header=main_radio_head,
                                                            keep_old_header=True, flux_conserve=False)
                fits.writeto(filename=out_dir + '/converted_radio/' +galaxy + '_' + radio_maps_ident[index] + '_regrid.fits',
                    data=radio_regrid_dat, header=radio_regrid_head, output_verify='silentfix', overwrite=True) 

        print('----------------- Done with this Galaxy')
    return 0

if __name__ == '__main__':

    print(os.getcwd())
    # make_thermal('n3044/n3044_input.csv',
    #             write_interim_files=True,with_lofar=True,target_res=20,
    #             Pypher_kernel='kernels/W4_to_20arcsec_gauss.fits',
    #             WISE_compare='example/ngc4013.w4.ss.fits',
    #             run_tag='n3044', plot_dir=plot_dir)
    make_thermal('example/ex_input.csv',
                write_interim_files=True,with_lofar=True,target_res=20,
                Pypher_kernel='kernels/W4_to_20arcsec_gauss.fits',
                WISE_compare='example/ngc4013.w4.ss.fits',
                run_tag='example', plot_dir=plot_dir)