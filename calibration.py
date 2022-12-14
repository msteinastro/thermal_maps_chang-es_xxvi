#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:03:19 2021

@author: michael
"""
# Flux Calibration
def halpha_flux_cal(data, factor):
    """
    Flux Calibration to [erg/s/cm/cm/pix]
    :param data:
    :param factor:
    :return:
    """
    print('Flux calibration Halpha data')
    print('Calibration factor: ', factor)
    cal = data * factor
    return cal


def wise_band4_cal(data, star_forming=True):
    """
    Flux Calibration to [Jy/pix]
    :param data:
    :param star_forming:
    :return:
    """

    print('Flux calibration WISE WERGA Band 4 data')
    # From README
    factor1 = 5.2269e-05  # DN to Jy
    factor2 = 0.97  # Aperture Correction
    factor3 = 1.0095  # Spectral Correction
    factor4 = 0.92  # Star forming
    if star_forming:
        factor = factor1 * factor2 * factor3 * factor4
    else:
        factor = factor1 * factor2 * factor3
    print('Calibration factor: ', factor)
    cal = data * factor
    return cal


