import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import os
from regions import read_ds9
from clip_regrid import clipping, regrid
from aux_functions import masked_array_to_filled_nan
work_dir = os.getcwd()  
plot_dir = work_dir + '/plots'

# ----------------------------------------------------------------------------------------------------------------------
# Plotting functions

def spec_index_plot(data_1, head_1, nu_1, data_2, head_2, nu_2, name, do_clip=False, do_regrid=False, write_fits=True,
                    min_val=None, max_val=None,
                    do_cut=False, sky_coord=None, size=None,
                    do_mask=False, mask_path=None,
                    cmap=None):
    wcs_1 = WCS(head_1)
    wcs_2 = WCS(head_2)
    if do_clip:
        print('Clip the data')
        data_1 = clipping(data=data_1, nsigma=3)
        data_2 = clipping(data=data_2, nsigma=3)
    if do_mask:
        print('Mask the data')
        ds9_reg = read_ds9(mask_path)
        ds9_reg_pix_1 = ds9_reg[0].to_pixel(wcs_1)
        ds9_reg_pix_2 = ds9_reg[0].to_pixel(wcs_2)
        mask_1 = ds9_reg_pix_1.to_mask().to_image(data_1.shape)
        mask_1_sky = np.invert(mask_1.astype('bool'))
        mask_2 = ds9_reg_pix_2.to_mask().to_image(data_2.shape)
        mask_2_sky = np.invert(mask_2.astype('bool'))
        data_1 = np.ma.masked_array(data=data_1, mask=mask_1_sky)
        data_2 = np.ma.masked_array(data=data_2, mask=mask_2_sky)
    if do_regrid:
        print('Regrid the data')
        pix_scale_1 = wcs.utils.proj_plane_pixel_scales(wcs_1)
        pix_scale_2 = wcs.utils.proj_plane_pixel_scales(wcs_2)
        size_1 = pix_scale_1[0]*pix_scale_1[1]
        size_2 = pix_scale_2[0]*pix_scale_2[1]
        if size_1 >= size_2:
            tar_dat, tar_head, tar_nu = data_1, head_1, nu_1
            regrid_dat, regrid_head, regird_nu = data_2, head_2, nu_2
        else:
            tar_dat, tar_head, tar_nu = data_2, head_2, nu_2
            regrid_dat, regrid_head, regird_nu = data_1, head_1, nu_1
        regrid_dat, regrid_head = regrid(data=regrid_dat, header=regrid_head, target_header=tar_head)
    else:
        tar_dat, tar_head, tar_nu = data_1, head_1, nu_1
        regrid_dat, regrid_head, regird_nu = data_2, head_2, nu_2
    tar_wcs = WCS(tar_head)
    spec_index = np.log10(tar_dat/regrid_dat)/np.log10(tar_nu/regird_nu)
    print('Spectral index map Min: %.2f, Max: %.2f' % (np.nanmin(spec_index), np.nanmax(spec_index)))

    name = 'spec_index/'+name
    print('Write file to:', name)
    wcs_plot(data=spec_index, head_wcs=tar_wcs, name=name, min_val=min_val, max_val=max_val,
             do_cut=do_cut, sky_coord=sky_coord, size=size, cmap=cmap)
    if write_fits:
        spec_index_fits = masked_array_to_filled_nan(spec_index)
        fits.writeto(filename=plot_dir + '/' + name + '.fits', data=spec_index_fits, header=tar_head, overwrite=True)

def spec_index_plot_new(data_1, head_1, nu_1, data_2, head_2, nu_2, name, do_clip=False, do_regrid=False,
                        write_fits=True, min_val=None, max_val=None, do_cut=False, sky_coord=None, size=None,
                        do_mask=False, mask_path=None, cmap=None):
    wcs_1 = WCS(head_1)
    wcs_2 = WCS(head_2)
    if do_clip:
        print('Clip the data')
        data_1 = clipping(data=data_1, nsigma=3)
        data_2 = clipping(data=data_2, nsigma=3)
    if do_mask:
        print('Mask the data')
        ds9_reg = read_ds9(mask_path)
        ds9_reg_pix_1 = ds9_reg[0].to_pixel(wcs_1)
        ds9_reg_pix_2 = ds9_reg[0].to_pixel(wcs_2)
        mask_1 = ds9_reg_pix_1.to_mask().to_image(data_1.shape)
        mask_1_sky = np.invert(mask_1.astype('bool'))
        mask_2 = ds9_reg_pix_2.to_mask().to_image(data_2.shape)
        mask_2_sky = np.invert(mask_2.astype('bool'))
        data_1 = np.ma.masked_array(data=data_1, mask=mask_1_sky)
        data_2 = np.ma.masked_array(data=data_2, mask=mask_2_sky)
    if do_regrid:
        print('Regrid the data')
        pix_scale_1 = wcs.utils.proj_plane_pixel_scales(wcs_1)
        pix_scale_2 = wcs.utils.proj_plane_pixel_scales(wcs_2)
        size_1 = pix_scale_1[0]*pix_scale_1[1]
        size_2 = pix_scale_2[0]*pix_scale_2[1]
        if size_1 >= size_2:
            tar_dat, tar_head, tar_nu = data_1, head_1, nu_1
            regrid_dat, regrid_head, regird_nu = data_2, head_2, nu_2
        else:
            tar_dat, tar_head, tar_nu = data_2, head_2, nu_2
            regrid_dat, regrid_head, regird_nu = data_1, head_1, nu_1
        regrid_dat, regrid_head = regrid(data=regrid_dat, header=regrid_head, target_header=tar_head)
    else:
        tar_dat, tar_head, tar_nu = data_1, head_1, nu_1
        regrid_dat, regrid_head, regird_nu = data_2, head_2, nu_2
    tar_wcs = WCS(tar_head)
    spec_index = np.log10(tar_dat/regrid_dat)/np.log10(tar_nu/regird_nu)
    print('Spectral index map Min: %.2f, Max: %.2f' % (np.nanmin(spec_index), np.nanmax(spec_index)))

    print('Write file to:', name)
    if min_val == np.nan:
        min_val = np.nanmin(spec_index)
    if max_val == np.nan:
        max_val == np.nanmax(spec_index)
    if do_cut:
        print('Cut the image at position:', sky_coord)
        print('Cut Size:', size)
        cut = Cutout2D(data=spec_index, wcs=tar_wcs, position=sky_coord, size=size)
        data = cut.data
        head_wcs = cut.wcs
    plt.subplot(projection=head_wcs)
    if cmap is not None:
        plt.imshow(data, vmin=min_val, vmax=max_val, origin='lower', cmap=cmap)
    else:
        plt.imshow(data, vmin=min_val, vmax=max_val, origin='lower')
    title = name.rsplit('/')[-1]
    plt.title(title)
    plt.grid()
    cax = plt.axes([0.82, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    print(f"Saving plot: {name}")
    plt.savefig(name, dpi=300)
    if write_fits:
        spec_index_fits = masked_array_to_filled_nan(spec_index)
        fits.writeto(filename=name + '.fits', data=spec_index_fits, header=tar_head, overwrite=True)

def point_source_removed_plot(data_1, head_1, data_2, head_2, name, do_clip=False, do_regrid=False,
                              min_val=None, max_val=None, do_cut=False, sky_coord=None, size=None, cmap=None):
    wcs_1 = WCS(head_1)
    wcs_2 = WCS(head_2)
    if do_clip:
        data_1 = clipping(data=data_1, nsigma=3)
        data_2 = clipping(data=data_2, nsigma=3)
    if do_regrid:
        pix_scale_1 = wcs.utils.proj_plane_pixel_scales(wcs_1)
        pix_scale_2 = wcs.utils.proj_plane_pixel_scales(wcs_2)
        size_1 = pix_scale_1[0]*pix_scale_1[1]
        size_2 = pix_scale_2[0]*pix_scale_2[1]
        if size_1 >= size_2:
            tar_dat, tar_head = data_1, head_1,
            regrid_dat, regrid_head = data_2, head_2
        else:
            tar_dat, tar_head = data_2, head_2
            regrid_dat, regrid_head = data_1, head_1
        regrid_dat, regrid_head = regrid(data=regrid_dat, header=regrid_head, target_header=tar_head)
    else:
        tar_dat, tar_head = data_1, head_1
        regrid_dat, regrid_head = data_2, head_2

    tar_wcs = WCS(tar_head)
    reg_wcs = WCS(regrid_head)
    if do_cut:
        print('Cut the image at position:', sky_coord)
        print('Cut Size:', size)
        cut_1 = Cutout2D(data=tar_dat, wcs=tar_wcs, position=sky_coord, size=size)
        tar_dat = cut_1.data
        tar_wcs = cut_1.wcs

        cut_2 = Cutout2D(data=regrid_dat, wcs=reg_wcs, position=sky_coord, size=size)
        regrid_dat = cut_2.data
        reg_wcs = cut_2.wcs

    #tar_dat = np.squeeze(tar_dat)
    #regrid_dat = np.squeeze(regrid_dat)
    mean_1, median_1, sig_1 = sigma_clipped_stats(tar_dat)
    mean_2, median_2, sig_2 = sigma_clipped_stats(regrid_dat)

    ratio = regrid_dat / tar_dat
    print('Ratio: Min: %.2f, Max: %.2f, Median: %.6f' % (np.nanmin(ratio), np.nanmax(ratio), np.nanmedian(ratio)))
    fig = plt.figure()#figsize=(15,3))
    #fig, axs = plt.subplots(1, 3)
    plt.subplot(221,projection=tar_wcs)
    plt.imshow(tar_dat, vmin=mean_1-3*sig_1, vmax=mean_1+5*sig_1)
    plt.colorbar()
    plt.title('Image')

    plt.subplot(222,projection=tar_wcs)
    plt.imshow(regrid_dat, vmin=mean_2-3*sig_2, vmax=mean_2+5*sig_2)
    plt.colorbar()
    plt.title('Point Sources Removed')

    plt.subplot(223,projection=tar_wcs)
    plt.imshow(ratio, vmax=1.5, vmin=-0.5, cmap='plasma')
    plt.colorbar()
    plt.contour(regrid_dat, levels=[mean_1+3*sig_1, mean_1+5*sig_1, mean_1+7*sig_1], colors='teal', linewidths=0.5)

    plt.title('Ratio')
    out_str = plot_dir + '/ps_comp/' + name
    fig.tight_layout(pad=4)
    plt.savefig(out_str, dpi=300)
    plt.close()


def wcs_plot(data, head_wcs, name, min_val=None, max_val=None,
             do_cut=False, sky_coord=None, size=None, cmap=None,
             with_text=False, text=None, plot_dir=plot_dir):
    print('Make WCS plot')
    if min_val == np.nan:
        min_val = np.nanmin(data)
    if max_val == np.nan:
        max_val == np.nanmax(data)
    if do_cut:
        print('Cut the image at position:', sky_coord)
        print('Cut Size:', size)
        cut = Cutout2D(data=data, wcs=head_wcs, position=sky_coord, size=size)
        data = cut.data
        head_wcs = cut.wcs
    plt.subplot(projection=head_wcs)
    if cmap is not None:
        plt.imshow(data, vmin=min_val, vmax=max_val, origin='lower', cmap=cmap)
    else:
        plt.imshow(data, vmin=min_val, vmax=max_val, origin='lower')
    if with_text:
        plt.annotate(text,(1,1))
        print('Added String to figure:', text)
    title = name.rsplit('/')[-1]
    plt.title(title)
    plt.grid()
    cax = plt.axes([0.82, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    out_str = plot_dir+'/'+name
    print(f"Saving plot: {name} in Folder: {plot_dir}")
    plt.savefig(out_str, dpi=300)

