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
# Import from my scripts:
import make_plots
work_dir = os.getcwd()  # '/home/michael/PycharmProjects/thermal_maps'
print('Current working directory:', work_dir)
plot_dir = work_dir + '/plots'
temp_dir = work_dir + '/temp'
out_dir = work_dir + '/outs'
# aux_map_dir = '/home/michael/phd_data/aux_maps/'
aux_map_dir = '/data/phd_data/aux_maps'
target_fwhm_arcsec = 15


# ----------------------------------------------------------------------------------------------------------------------
# Auxiliary Functions:
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
    date = datetime.now().strftime('%m/%d/%y')
    header['COMMENT'] = 'File written on ' + date + ' by M. Stein (AIRUB)'


def fill_nans(infile, outfile, fill_value=0.0, hdu=0):
    fits_file = fits.open(infile)
    data = fits_file[hdu].data
    header = fits_file[hdu].header
    data_filled = np.nan_to_num(data, nan=fill_value)
    fits.writeto(outfile, data=data_filled, header=header)

def clip_fits(infile, outfile, clip_val, hdu=0):
    data, header = load_fits(infile, header_ext=hdu)
    header['COMMENT'] = 'Clipped at: ' + str(clip_val)
    data_clip = clip_value(data=data, value=clip_val)
    fits.writeto(outfile, data=data, header=header, overwrite=True)
    return


def read_setup_file(setup_path):
    """
    Reads the setup and parses all required arguments to the following functions
    """
    df_params = pd.read_csv(setup_path, header=0)
    print('Loaded Parameter File:')
    print(df_params)
    return df_params


def masked_array_to_filled_nan(masked_array):
    if isinstance(masked_array, np.ma.MaskedArray):
        filled_array = masked_array.filled(fill_value=np.nan)
    else:
        filled_array = masked_array
    return filled_array


# ----------------------------------------------------------------------------------------------------------------------
# Conversions:
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


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# IR weighting for Halpha correction:
a_old = 0.031
a_new = 0.042

nu_wise4 = wavelength_to_freq(22e-6)  # 22 micron data (w4 band)
nu_24mu = wavelength_to_freq(24e-6)
nuGHzC = 5.99  # freq of C band in GHz
nuGHzL = 1.5   # L band
nuMHzLoTTS = 144  # MHz
# Extinction correction from Vargas 2018:
extinction_factor = 1.36


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# PSF Convolution
#WISE_convolution_path = 'aux_files/w4cut_to_15arcsec_head_edit.fits'
WISE_convolution_path = 'aux_files/W4_to_15arcsec_gauss.fits'
#WISE_convolution_fits = fits.open(WISE_convolution_path)
#WISE_convolution_data = WISE_convolution_fits[0].data
#WISE_convolution_header = WISE_convolution_fits[0].header
#WISE_compare = '/home/michael/phd_data/aux_maps/WISE-WERGA/skysubtracted/22micron/ngc2683.w4.ss.fits'

WISE_comment = 'Data convolved to 15 arcsec Gauss PSF using a Pypher Kernel'
Halpha_comment = 'Data convolved to 15 arcsec Gauss PSF using a purely Gaussian Kernel'


def make_halpha_kernel(header, fwhm_arcsec):
    w_halpha = WCS(header)
    scale = wcs.utils.proj_plane_pixel_scales(w_halpha)
    mean_scale = np.mean(scale) * u.degree
    scale_arcsec = mean_scale.to(u.arcsec)
    fwhm_pixel = fwhm_arcsec/scale_arcsec.value
    std_pixel = fwhm_to_stddev(fwhm_pixel)
    kernel = Gaussian2DKernel(std_pixel)
    return kernel


def down_sample_kernel(psf_kernel, wise_compare, manual=False, down_block=(0, 0)): # psf_kernel=Pypher_kernel
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


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# Make Thermal maps
def make_thermal(setup_file, do_clip=False, nsig_alpha=3, nsig_wise=3,
                 do_mask=True, do_cal=True, do_convolution=True, do_regrid_halpha=True,
                 do_regird_to_changes=True, reduce_size=False, target_res=15, extinction_correction=True,
                 write_interim_files=False, write_non_thermal=True,
                 show_plots=False, halpha_in_erg_s=False, with_lofar=False, run_tag=None,
                 out_dir=out_dir, plot_dir=plot_dir,
                 Pypher_kernel='/home/michael/PycharmProjects/thermal_maps/outs/w4cut_to_15arcsec.fits',
                 WISE_compare='/data/phd_data/aux_maps/WISE-WERGA/skysubtracted/22micron/ngc2683.w4.ss.fits'):
    if run_tag is not None:
        out_dir = out_dir + '/' + run_tag
        plot_dir = plot_dir + '/' + run_tag
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
        changes = row['changes_path']
        mask = row['mask_path']
        if with_lofar:
            lofar = row['lofar_path']
        # Load fits files
        halpha_dat, halpha_head = load_fits(fits_path=halpha, fix_wcs=True)
        w_halpha = WCS(halpha_head)
        wise_dat, wise_head = load_fits(fits_path=wise, fix_wcs=True)
        w_wise = WCS(wise_head)
        changes_dat, changes_head = load_fits(fits_path=changes)
        changes_filename = changes.split('/')[-1]
        changes_filename_clean = changes_filename.split('.')[0]
        if with_lofar:
            print('Load LOFAR data:')
            lofar_dat, lofar_head = load_fits(fits_path=lofar)
            lofar_filename = lofar.split('/')[-1]
            lofar_filename_clean = lofar_filename.split('.')[0]
            print('Regrid LOFAR data to CHANG-ES frame')
            # Jy/Beam map: do not conserve sum of pixel values!
            lofar_regrid_dat, lofar_regrid_head = regrid(data=lofar_dat, header=lofar_head, target_header=changes_head,
                                                         keep_old_header=True, flux_conserve=False)
            if write_interim_files:
                fits.writeto(filename=out_dir + '/' +galaxy + '_lofar_regrid.fits',
                             data=lofar_regrid_dat, header=lofar_regrid_head, output_verify='fix', overwrite=True)
        print('Loaded  all data')
        print('Max Halpha', np.nanmax(halpha_dat))
        print('Max WISE', np.nanmax(wise_dat))
        # Cut off two extra wcs axes and remove unnecessary keys
        """
        changes_dat = np.squeeze(changes_dat)
        print('Changes map shape after squeezing:', changes_dat.shape)
        changes_head = edit_changes_header(changes_head)
        w_changes = WCS(changes_head)
        changes_head.update(w_changes.to_header())
        fits.writeto(filename=temp_dir+'/changes_2d.fits',
                     data=changes_dat, header=changes_head, output_verify='fix', overwrite=True)
        print('Header after updating wcs')
        print(repr(changes_head))
        """
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
            if extinction_correction:
                print('Perform extinction correction, with factor: ', extinction_factor)
                wise_dat_cal = wise_dat_cal * extinction_factor
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
            fits.writeto(filename=temp_dir + '/ir24mu_after_cal.fits', data=ir_24_mu_erg_s_cm_cm_filled,
                         overwrite=True)
            fits.writeto(filename=temp_dir + '/halpha_after_cal.fits', data=halpha_dat_filled, overwrite=True)
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
                fits.writeto(filename=temp_dir + '/ir24mu_after_conv.fits', data=ir_24_mu_erg_s_cm_cm_filled,
                             overwrite=True)
                fits.writeto(filename=temp_dir + '/halpha_after_conv.fits', data=halpha_dat_filled, overwrite=True)

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
            fits.writeto(filename=temp_dir+'/ir24mu_erg_s.fits', data=ir_24_mu_erg_s_filled, overwrite=True)
            fits.writeto(filename=temp_dir+'/halpha_erg_s.fits', data=halpha_erg_s_filled, overwrite=True)

        print('Brightest Pixel H alpha', np.nanmax(halpha_erg_s))
        halpha_corr_mix = halpha_erg_s+(a_new*ir_24_mu_erg_s)
        print('Halpha corrected')
        print('Max Halpha_corr', np.nanmax(halpha_corr_mix))
        if write_interim_files:
            correction_diff = (halpha_corr_mix - halpha_erg_s)/halpha_corr_mix
            fits.writeto(filename=temp_dir+'/rel_halpha_corr_halpha_diff.fits', data=correction_diff, overwrite=True)
        sfr_map = sfr_murphy_2011(halpha_corr_mix)
        sfr_head = wise_head.copy()
        sfr_head['BUNIT'] = 'M_sol/year'
        fits.writeto(filename=out_dir+'/'+galaxy+'sfr_map.fits',
                     data=sfr_map, header=sfr_head,overwrite=True)
        l_thermal_lband = thermal_murphy_2011(t_electron=1e4, freq_ghz=nuGHzL, sfr=sfr_map)
        l_thermal_cband = thermal_murphy_2011(t_electron=1e4, freq_ghz=nuGHzC, sfr=sfr_map)
        l_thermal_lotss = thermal_murphy_2011(t_electron=1e4, freq_ghz=nuMHzLoTTS*1e3, sfr=sfr_map)
        thermal_lband_jy = erg_s_cm_cm_hz_to_jy(l_thermal_lband, distance_sphere=luminosity_sphere)
        thermal_cband_jy = erg_s_cm_cm_hz_to_jy(l_thermal_cband, distance_sphere=luminosity_sphere)
        thermal_lotss_jy = erg_s_cm_cm_hz_to_jy(l_thermal_lotss, distance_sphere=luminosity_sphere)
        halpha_head_regrid['BUNIT'] = 'Jy'
        date = datetime.now().strftime('%m/%d/%y')
        halpha_head_regrid['COMMENT'] = 'File written on ' + date + ' by M. Stein (AIRUB)'

        if write_interim_files:
            fits.writeto(filename=out_dir+'/thermal_maps/'+galaxy+'_thermal_Lband.fits',
                         data=thermal_lband_jy, header=halpha_head_regrid, overwrite=True, output_verify='silentfix')
            fits.writeto(filename=out_dir + '/thermal_maps/' + galaxy + '_thermal_Cband.fits',
                         data=thermal_cband_jy, header=halpha_head_regrid, overwrite=True, output_verify='silentfix')
            fits.writeto(filename=out_dir + '/thermal_maps/' + galaxy + '_thermal_LoTSS.fits',
                         data=thermal_lotss_jy, header=halpha_head_regrid, overwrite=True, output_verify='silentfix')
        if do_regird_to_changes:
            print('Reproject thermal maps to CHANG-ES map')
            thermal_lband_jy_repr, head_lband = regrid(data=thermal_lband_jy, header=halpha_head_regrid,
                                                   target_header=changes_head, keep_old_header=True)
            thermal_cband_jy_repr, head_cband = regrid(data=thermal_cband_jy, header=halpha_head_regrid,
                                                   target_header=changes_head, keep_old_header=True)
            thermal_lotss_jy_repr, head_lotss = regrid(data=thermal_lotss_jy, header=halpha_head_regrid,
                                                   target_header=changes_head, keep_old_header=True)
            if write_interim_files:
                print(str('Write thermal maps to ' + out_dir + '/thermal_maps/'))
                fits.writeto(filename=out_dir+'/thermal_maps/'+galaxy+'_thermal_Lband_repr.fits',
                             data=thermal_lband_jy_repr, header=changes_head, overwrite=True,
                             output_verify='fix')
                fits.writeto(filename=out_dir + '/thermal_maps/' + galaxy + '_thermal_Cband_repr.fits',
                             data=thermal_cband_jy_repr, header=changes_head, overwrite=True,
                             output_verify='fix')
                fits.writeto(filename=out_dir + '/thermal_maps/' + galaxy + '_thermal_LoTSS_repr.fits',
                             data=thermal_lotss_jy_repr, header=changes_head, overwrite=True,
                             output_verify='fix')
        print('Convert Changes Data to Jy/pix')
        changes_dat_jy, changes_head_jy = jy_beam_to_jy(data=changes_dat, header=changes_head)
        fits.writeto(filename=out_dir + '/converted_radio/' + changes_filename_clean + '_jy_pix.fits',
                     data=changes_dat_jy, header=changes_head_jy, overwrite=True)
        thermal_frac_lband = thermal_lband_jy_repr / changes_dat_jy
        clipped_mean = sigma_clipped_stats(thermal_frac_lband)[0]
        print('Average thermal fraction Lband:', clipped_mean)
        figure_string = 'Clipped mean thermal fraction: ' + "{:.1f}".format(100 * clipped_mean) + '%'
        make_plots.wcs_plot(data=thermal_frac_lband, head_wcs=WCS(changes_head_jy),
                            name=galaxy + '_thermal_frac_lband',
                            min_val=0, max_val=min(0.5, np.nanmax(thermal_lband_jy_repr / changes_dat_jy)), cmap='plasma',
                            do_cut=True, sky_coord=SkyCoord(coord), size=size * u.arcmin,
                            with_text=True, text=figure_string, plot_dir=plot_dir)
        if write_non_thermal:
            print(str('Write Non Thermal Changes map to ' + out_dir + '/thermal_maps/'))
            #Non Thermal map for the lband
            thermal_lband_jy_repr_filled = np.nan_to_num(thermal_lband_jy_repr)
            changes_non_thermal = changes_dat_jy - thermal_lband_jy_repr_filled
            fits.writeto(filename=out_dir + '/thermal_maps/' + changes_filename_clean + '_non_thermal_jy_pix.fits',
                         data=changes_non_thermal, header=changes_head_jy, overwrite=True)
            changes_nt_jybeam_dat, changes_nt_jybeam_head = jy_to_jy_beam(data=changes_non_thermal ,header=changes_head_jy)
            fits.writeto(filename=out_dir + '/thermal_maps/' + changes_filename_clean + '_non_thermal_jy_beam.fits',
                         data=changes_nt_jybeam_dat, header=changes_nt_jybeam_head, overwrite=True)
        if with_lofar:
            print('Convert LOFAR Data to Jy/pix')
            lofar_dat_jy, lofar_head_jy = jy_beam_to_jy(data=lofar_regrid_dat, header=lofar_regrid_head)
            fits.writeto(filename=out_dir + '/converted_radio/' + lofar_filename_clean + '_jy_pix.fits',
                         data=lofar_dat_jy, header=lofar_head_jy, overwrite=True)
            thermal_frac_lotss = thermal_lotss_jy_repr / lofar_dat_jy
            clipped_mean = sigma_clipped_stats(thermal_frac_lotss)[0]
            print('Average thermal fraction LoTSS:', clipped_mean)
            figure_string = 'Clipped mean thermal fraction: ' + "{:.1f}".format(100* clipped_mean)+ '%'
            make_plots.wcs_plot(data=thermal_frac_lotss, head_wcs=WCS(changes_head_jy),
                                name=galaxy + '_thermal_frac_lotss',
                                min_val=0, max_val=min(0.1, np.nanmax(thermal_lotss_jy_repr / lofar_dat_jy)), cmap='plasma',
                                do_cut=True, sky_coord=SkyCoord(coord), size=size * u.arcmin,
                                with_text=True, text=figure_string, plot_dir=plot_dir)
            if write_non_thermal:
                print(str('Write Non Thermal LOFAR map to ' + out_dir + '/thermal_maps/'))
                # Non Thermal map for the lofar
                thermal_lotss_jy_repr_filled = np.nan_to_num(thermal_lotss_jy_repr)
                lofar_non_thermal = lofar_dat_jy - thermal_lotss_jy_repr_filled
                fits.writeto(filename=out_dir + '/thermal_maps/' + lofar_filename_clean + '_non_thermal_jy_pix.fits',
                             data=lofar_non_thermal, header=lofar_head_jy, overwrite=True)
                lofar_nt_jybeam_dat, lofar_nt_jybeam_head = jy_to_jy_beam(data=lofar_non_thermal,
                                                                              header=lofar_head_jy)
                fits.writeto(filename=out_dir + '/thermal_maps/' + lofar_filename_clean + '_non_thermal_jy_beam.fits',
                             data=lofar_nt_jybeam_dat, header=lofar_nt_jybeam_head, overwrite=True)
        print('----------------- Done with this Galaxy')
    return 0



if __name__ == '__main__':
    # Pypher_kernel = '/home/michael/PycharmProjects/thermal_maps/outs/w4cut_to_15arcsec.fits'
    WISE_convolution_fits = fits.open(WISE_convolution_path)
    WISE_convolution_data = WISE_convolution_fits[0].data
    WISE_convolution_header = WISE_convolution_fits[0].header
    os.chdir('/data/phd_data')
    print('Data Directory:', os.getcwd())
    make_thermal(work_dir+'/test_param', write_interim_files=True,with_lofar=True,
                 Pypher_kernel='/home/michael/PycharmProjects/thermal_maps/aux_files/W4_to_15arcsec_gauss.fits',
                 run_tag='test_run')
    #make_thermal('test_param_ng891_carlos', do_clip=False, do_mask=False, do_cal=False, do_convolution=False,
                #do_regrid_halpha=False, do_regird_to_changes=False, write_interim_files=True, halpha_in_erg_s=True)
