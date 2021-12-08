#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:04:42 2021

@author: michael
"""
# Python enviroment imports
import astropy.wcs as wcs
from astropy.wcs import WCS
import numpy as np
from radio_beam import Beam
from scipy import constants

# Local imports
from aux_functions import date_to_header


def parsec_to_cm(x):
    return 3.086e+18 * x


def fwhm_to_stddev(x):
    sig = 1 / (2 * np.sqrt(2 * np.log(2))) * x
    return sig


def stddev_to_fwhm(x):
    fwhm = 2 * np.sqrt(2 * np.log(2)) * x
    return fwhm


def jy_to_erg_s_cm_cm(jy, frequency):
    erg_s_cm_cm = jy * 1e-23 * frequency
    return erg_s_cm_cm


def jy_beam_to_jy(data, get_from_header=True, header=None, pix_size=0, beam_fwhm=0):
    """
    Function to convert from Jy/Beam to Jy/pix.
    :param data: data to convert.
    :param get_from_header: If True, you need to pass a fits header.
    :param header: fits header.
    :param pix_size: pixel size in deg.
    :param beam_fwhm: beam FWHM in deg.
    :return: If get_from header==True: returns corrected data and edited header, else: returns corrected data.
    """
    if get_from_header:
        beam = Beam.from_fits_header(header)
        w = WCS(header)
        pix_dimensions = wcs.utils.proj_plane_pixel_scales(w)
        pix_size_head = pix_dimensions[0] * pix_dimensions[1]
        beam_size = beam.major * beam.minor
        correction_factor = 1/(1.1331 * (beam_size.value/pix_size_head)) # 1.11331 = pi/(4*ln(2)
        print('Correction factor to convert to Jy:', correction_factor)
        header['BUNIT'] = 'Jy'
        date_to_header(header)
        data_corr = data * correction_factor
        return data_corr, header
    else:
        correction_factor = 1/(1.1331 * beam_fwhm**2/pix_size**2)
        print('Correction factor to convert to Jy:', correction_factor)
        data_corr = data * correction_factor
        return data_corr

def jy_to_jy_beam(data, get_from_header=True, header=None, pix_size=0, beam_fwhm=0):
    """
    Function to convert from  Jy/pix to Jy/Beam.
    :param data: data to convert.
    :param get_from_header: If True, you need to pass a fits header.
    :param header: fits header.
    :param pix_size: pixel size in deg.
    :param beam_fwhm: beam FWHM in deg.
    :return: If get_from header==True: returns corrected data and edited header, else: returns corrected data.
    """
    if get_from_header:
        beam = Beam.from_fits_header(header)
        w = WCS(header)
        pix_dimensions = wcs.utils.proj_plane_pixel_scales(w)
        pix_size_head = pix_dimensions[0] * pix_dimensions[1]
        beam_size = beam.major * beam.minor
        correction_factor = (1.1331 * (beam_size.value/pix_size_head)) # 1.11331 = pi/(4*ln(2)
        print('Correction factor to convert to Jy:', correction_factor)
        header['BUNIT'] = 'Jy/Beam'
        date_to_header(header)
        data_corr = data * correction_factor
        return data_corr, header
    else:
        correction_factor = (1.1331 * beam_fwhm**2/pix_size**2)
        print('Correction factor to convert to Jy:', correction_factor)
        data_corr = data * correction_factor
        return data_corr

def erg_s_cm_cm_hz_to_jy(spec_lum, distance_sphere):
    jy = spec_lum * 1e23/distance_sphere
    return jy


def wavelength_to_freq(wl):
    f = constants.speed_of_light / wl
    return f

# Astrophysical Relations
def sfr_murphy_2011(halpha):
    """
    Computes the star formation given a halpha map following Murphy et al. 2011
    :param halpha:
    :return:
    """
    sfr = 5.37e-42 * halpha
    return sfr


def thermal_murphy_2011(t_electron, freq_ghz, sfr):
    """
    Computes the thermal emission given a star formation map following Murphy et al. 2011 Eq 11
    :param t_electron:
    :param freq_ghz:
    :param sfr: map of the star formation in M_sun/yr
    :return:
    """
    thermal_emission = 2.2e27 * ((t_electron/1e4)**0.45) * (freq_ghz ** -0.1) * sfr
    return thermal_emission