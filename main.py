#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
