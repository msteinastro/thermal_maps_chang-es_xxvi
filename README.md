# CHANG-ES. XXVI.
## Insights into cosmic-ray transport from radio halos in edge-on galaxies

The code that is presented here, has been developed as part of a peer reviewed article that is accepted for publication in Astronomy & Astrophysics. 

NASA ADS: https://ui.adsabs.harvard.edu/abs/2022arXiv221007709S/abstract

DOI: 10.1051/0004-6361/202243906

If you find the code useful for your own project, please cite the paper and link to this github repository.

final version will be pusbslished with the accepted publication.
### Files:
The code ist structered in multiple files:

- aux_functions.py: small functions to edit arrays and handle files or datatypes.
- calibration.py: calibration functions to calibrate the halpha (to [erg/s/cm/cm/pix]) and the WISE band 4 data ([Jy/pix]). This is only valid for data that comes from the internal CHANG-ES Server. **Make sure that this calibration is valid for your data!**
- convolution.py: Everything connected to convolving data.
- thermal_maps.py: The main routine to create thermal maps and subtract the from radio maps.
- math_physics.py: Conversion for the required physical units as well as the star formation (SFR) and thermal emission estimates from Murphy+2011.

### Preparing a homogenization kernel:
The WISE band 4 (22micron) PSF is far from beeing gaussian. Therefore, to convolve to a common gaussian beam we use pypher (https://pypher.readthedocs.io/en/latest/) to create an homogenization kernel. 
The folder kernels contains the WISE band 4 PSF (empirical PSF created by stacking stars, thanks to T. Jarrett for providing the PSF), a 15'' and 20'' gaussian PSF as well as the homogenization kernels to go from the WISE band 4 PSF to 15'' or 20'' gaussian kernels. 

You also can create your own kernels by downloading and installing pyhper. Then just run:

`pypher W4_PSF_cut.fits your_kernel.fits output_homogenization_kernel.fits`

### The main routine **make_thermal**:
The code is designed to be executed on several galaxies in one run. Galaxy specific parameters are passed via an input csv file. This file needs to contain the following columns:
- galaxy: Name tag used for writing out files.
- distance[Mpc]: Distance to the glaxy that is used to compute the luminosity.
- coord: RA Dec stored as string (e.g.: 02h22m33.41s+42d20m56.9s).
- size[arcmin]: Diameter of the galaxy.
- halpha_flux_cal: Flux calibration factor to calibrate the H-alpha data to [erg/s/cm/cm/pix].
- wise_path: Path to Wise file already calibrated to Jy/Pix.
- halpha_path: Path to Halpha File.
- changes_path: Path to 1.5 GHz Data. 
- lofar_path: Path to 144 MHz Data.
- mask_path: Path to DS9 Region file that is used for aperture flux measurements.

When calling the `make_thermal` function, you need to specify some extra parameters that are not galaxy specific:
- Path to the setup file.
- Common resolution of the radio maps (needed for convolving the H-alpha data)

You can also turn several features on and off to enabable or disable several features. Just have a look at the implementaion of `make_thermal` in main.py

There is an example that should run when executing `thermal_maps.py`

In the branch as_published, you can find barebone code that has been used, in the original publication. In the main branch, the code as been restructured for better usability.
