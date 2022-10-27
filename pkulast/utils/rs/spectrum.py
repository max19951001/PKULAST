# -*- coding:utf-8 -*-
# Copyright (c) 2021-2022.

################################################################
# The contents of this file are subject to the GPLv3 License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# https://www.gnu.org/licenses/gpl-3.0.en.html

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PKULAST python package.

# Initial Dev of the Original Code is Jinshun Zhu, PhD Student,
# Institute of Remote Sensing and Geographic Information System,
# Peking Universiy Copyright (C) 2022
# All Rights Reserved.

# Contributor(s): Jinshun Zhu (created, refactored and updated original code).
###############################################################

import os
import logging
import numpy as np
from scipy import constants
from pkulast.config import Modtran_Sampling_Resolution


LOG = logging.getLogger(__name__)

def get_mmd(emission):
    beta = emission / np.mean(emission)
    return np.max(beta) - np.min(beta)

def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    b = np.mean(ys) - m * np.mean(xs)
    return m, b

def RATIO(emiss):
    return emiss / np.mean(emiss, axis=1)[:, np.newaxis]

def bt2r(band, bt):
    ''' Convert bright temperature to radiance
	'''
    return band.bt2r(bt)

def r2bt(band, radiance):
    """ Convert radiance to bright temperature
	"""
    return band.r2bt(radiance)

def spectral_interplotion(f, spectral_wv, spectral_res):
    ''' spectral to band interpolation
	'''
    band_interp = []
    for band in f:
        band_interp.append( np.sum(np.interp(band.wv, spectral_wv, spectral_res) * band.response) / np.sum(band.response) )
    return band_interp

def convert2wavenumber(rsr):
    """Convert Spectral Responses from wavelength to wavenumber space.

    Take rsr data set with all channels and detectors for an instrument
    each with a set of wavelengths and normalised responses and
    convert to wavenumbers and responses

    :rsr: Relative Spectral Response function (all bands)
    Returns:
      :retv: Relative Spectral Responses in wave number space
      :info: Dictionary with scale (to go convert to SI units) and unit

    """
    retv = {}
    for chname in rsr.keys():  # Go through bands/channels
        retv[chname] = {}
        for det in rsr[chname].keys():  # Go through detectors
            retv[chname][det] = {}
            if 'wavenumber' in rsr[chname][det].keys():
                # Make a copy. Data are already in wave number space
                retv[chname][det] = rsr[chname][det].copy()
                LOG.debug("RSR data already in wavenumber space. No conversion needed.")
                continue

            for sat in rsr[chname][det].keys():
                if sat == "wavelength":
                    # micro meters to cm
                    wnum = 1. / (1e-4 * rsr[chname][det][sat])
                    retv[chname][det]['wavenumber'] = wnum[::-1]
                elif sat == "response":
                    # Flip the response array:
                    if isinstance(rsr[chname][det][sat], dict):
                        retv[chname][det][sat] = {}
                        for name in rsr[chname][det][sat].keys():
                            resp = rsr[chname][det][sat][name]
                            retv[chname][det][sat][name] = resp[::-1]
                    else:
                        resp = rsr[chname][det][sat]
                        retv[chname][det][sat] = resp[::-1]

    unit = 'cm-1'
    si_scale = 100.0
    return retv, {'unit': unit, 'si_scale': si_scale}


def get_central_wave(wav, resp, weight=1.0):
    """Calculate the central wavelength or the central wavenumber.

    Calculate the central wavelength or the central wavenumber, depending on
    which parameters is input.  On default the weighting funcion is
    f(lambda)=1.0, but it is possible to add a custom weight, e.g. f(lambda) =
    1./lambda**4 for Rayleigh scattering calculations

    """
    # info: {'unit': unit, 'si_scale': si_scale}
    # To get the wavelenght/wavenumber in SI units (m or m-1):
    # wav = wav * info['si_scale']

    # res = np.trapz(resp*wav, wav) / np.trapz(resp, wav)
    # Check if it is a wavelength or a wavenumber and convert to microns or cm-1:
    # This should perhaps be user defined!?
    # if info['unit'].find('-1') > 0:
    # Wavenumber:
    #     res *=
    return np.trapz(resp * wav * weight, wav) / np.trapz(resp * weight, wav)

def convert2hdf5(ClassIn, platform_name, bandnames, scale=1e-06):
    """Retrieve original RSR data and convert to internal hdf5 format.

    *scale* is the number which has to be multiplied to the wavelength data in
    order to get it in the SI unit meter

    """
    import h5py

    instr = ClassIn(bandnames[0], platform_name)
    instr_name = instr.instrument.replace('/', '')
    filename = os.path.join(instr.output_dir,
                            "rsr_{0}_{1}.h5".format(instr_name,
                                                    platform_name))

    with h5py.File(filename, "w") as h5f:
        h5f.attrs['description'] = ('Relative Spectral Responses for ' +
                                    instr.instrument.upper())
        h5f.attrs['platform_name'] = platform_name
        h5f.attrs['band_names'] = bandnames

        for chname in bandnames:
            sensor = ClassIn(chname, platform_name)
            grp = h5f.create_group(chname)
            wvl = sensor.rsr['wavelength'][~np.isnan(sensor.rsr['wavelength'])]
            rsp = sensor.rsr['response'][~np.isnan(sensor.rsr['wavelength'])]
            grp.attrs['central_wavelength'] = get_central_wave(wvl, rsp)
            arr = sensor.rsr['wavelength']
            dset = grp.create_dataset('wavelength', arr.shape, dtype='f')
            dset.attrs['unit'] = 'm'
            dset.attrs['scale'] = scale
            dset[...] = arr
            arr = sensor.rsr['response']
            dset = grp.create_dataset('response', arr.shape, dtype='f')
            dset[...] = arr


def get_effective_quantity(spectraldomain,  spectral_quantity,  spectral_baseline):
    """Normalise a spectral quantity to a scalar, using a weighted mapping by another spectral quantity.

    Effectivevalue =  integral(spectral_quantity * spectral_baseline) / integral( spectral_baseline)

    The data in spectral_quantity and  spectral_baseline must both be sampled at the same
    domain values  as specified in spectraldomain.

    The integral is calculated with numpy/scipy trapz trapezoidal integration function.

    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength, frequency or wavenumber.
        | spectral_quantity (np.array[N,] or [N,1]):  spectral quantity to be normalised
        | spectral_baseline (np.array[N,] or [N,1]):  spectral serving as baseline for normalisation

    Returns:
        | (float):  effective value
        | Returns None if there is a problem

    Raises:
        | No exception is raised.
    """

    num=np.trapz(spectral_quantity.reshape(-1, 1) * spectral_baseline.reshape(-1, 1), spectraldomain, axis=0)[0]
    den=np.trapz(spectral_baseline.reshape(-1, 1), spectraldomain, axis=0)[0]
    return num/den


def convert_spectral_domain(inspectraldomain,  type=''):
    """Convert spectral domains, i.e. between wavelength [um], wavenummber [cm^-1] and frequency [Hz]

    In string variable type, the 'from' domain and 'to' domains are indicated each with a single letter:
    'f' for temporal frequency, 'l' for wavelength and 'n' for wavenumber
    The 'from' domain is the first letter and the 'to' domain the second letter.

    Note that the 'to' domain vector is a direct conversion of the 'from' domain
    to the 'to' domain (not interpolated or otherwise sampled.

    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength, frequency or wavenumber.
        |    wavelength vector in  [um]
        |    frequency vector in  [Hz]
        |    wavenumber vector in   [cm^-1]
        | type (string):  specify from and to domains:
        |    'lf' convert from wavelength to per frequency
        |    'ln' convert from wavelength to per wavenumber
        |    'fl' convert from frequency to per wavelength
        |    'fn' convert from frequency to per wavenumber
        |    'nl' convert from wavenumber to per wavelength
        |    'nf' convert from wavenumber to per frequency

    Returns:
        | [N,1]: outspectraldomain
        | Returns zero length array if type is illegal, i.e. not one of the expected values

    Raises:
        | No exception is raised.
    """

    #use dictionary to switch between options, lambda fn to calculate, default zero
    outspectraldomain = {
              'lf': lambda inspectraldomain:  constants.c / (inspectraldomain * 1.0e-6),
              'ln': lambda inspectraldomain:  (1.0e4/inspectraldomain),
              'fl': lambda inspectraldomain:  constants.c  / (inspectraldomain * 1.0e-6),
              'fn': lambda inspectraldomain:  (inspectraldomain / 100) / constants.c ,
              'nl': lambda inspectraldomain:  (1.0e4/inspectraldomain),
              'nf': lambda inspectraldomain:  (inspectraldomain * 100) * constants.c,
              }.get(type, lambda inspectraldomain: np.zeros(shape=(0, 0)) )(inspectraldomain)

    return outspectraldomain


def convert_spectral_density(inspectraldomain,  inspectralquantity, type=''):
    """Convert spectral density quantities, i.e. between W/(m^2.um), W/(m^2.cm^-1) and W/(m^2.Hz).

    In string variable type, the 'from' domain and 'to' domains are indicated each with a
    single letter:
    'f' for temporal frequency, 'w' for wavelength and ''n' for wavenumber
    The 'from' domain is the first letter and the 'to' domain the second letter.

    The return values from this function are always positive, i.e. not mathematically correct,
    but positive in the sense of radiance density.

    The spectral density quantity input is given as a two vectors: the domain value vector
    and the density quantity vector. The output of the function is also two vectors, i.e.
    the 'to' domain value vector and the 'to' spectral density. Note that the 'to' domain
    vector is a direct conversion of the 'from' domain to the 'to' domain (not interpolated
    or otherwise sampled).

    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength,
            frequency or wavenumber.
        | inspectralquantity (np.array[N,] or [N,1]):  spectral density in same domain
           as domain vector above.
        |    wavelength vector in  [um]
        |    frequency vector in  [Hz]
        |    wavenumber vector in   [cm^-1]
        | type (string):  specify from and to domains:
        |    'lf' convert from per wavelength interval density to per frequency interval density
        |    'ln' convert from per wavelength interval density to per wavenumber interval density
        |    'fl' convert from per frequency interval density to per wavelength interval density
        |    'fn' convert from per frequency interval density to per wavenumber interval density
        |    'nl' convert from per wavenumber interval density to per wavelength interval density
        |    'nf' convert from per wavenumber interval density to per frequency interval density

    Returns:
        | ([N,1],[N,1]): outspectraldomain and outspectralquantity
        | Returns zero length arrays is type is illegal, i.e. not one of the expected values

    Raises:
        | No exception is raised.
    """

    inspectraldomain = inspectraldomain.reshape(-1,)
    inspectralquantity = inspectralquantity.reshape(inspectraldomain.shape[0], -1)
    outspectralquantity = np.zeros(inspectralquantity.shape)

    # the meshgrid idea does not work well here, because we can have very long
    # spectral arrays and these become too large for meshgrid -> size **2
    # we have to loop this one
    spec = inspectraldomain
    for col in range(inspectralquantity.shape[1]):

        quant = inspectralquantity[:,col]

        #use dictionary to switch between options, lambda fn to calculate, default zero
        outspectraldomain = {
                  'lf': lambda spec:  constants.c / (spec * 1.0e-6),
                  'fn': lambda spec:  (spec / 100) / constants.c ,
                  'nl': lambda spec:  (1.0e4/spec),
                  'ln': lambda spec:  (1.0e4/spec),
                  'nf': lambda spec:  (spec * 100) * constants.c,
                  'fl': lambda spec:  constants.c  / (spec * 1.0e-6),
                  }.get(type, lambda spec: np.zeros(shape=(0, 0)) )(spec)

        outspectralquantity[:, col] = {
                  'lf': lambda quant: quant / (constants.c *1.0e-6 / ((spec * 1.0e-6)**2)),
                  'fn': lambda quant: quant * (100 *constants.c),
                  'nl': lambda quant: quant / (1.0e4 / spec**2) ,
                  'ln': lambda quant: quant / (1.0e4 / spec**2) ,
                  'nf': lambda quant: quant / (100 * constants.c),
                  'fl': lambda quant: quant / (constants.c *1.0e-6 / ((spec * 1.0e-6)**2)),
                  }.get(type, lambda quant: np.zeros(shape=(0, 0)) )(quant)

    return (outspectraldomain,outspectralquantity)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    Source: http://wiki.scipy.org/Cookbook/SavitzkyGolay

    The Savitzky Golay filter is a particular type of low-pass filter,
    well adapted for data smoothing. For further information see:
    http://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf


    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.


    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples:
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()

    References:
        [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
            Data by Simplified Least Squares Procedures. Analytical
            Chemistry, 1964, 36 (8), pp 1627-1639.
        [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
            W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
            Cambridge University Press ISBN-13: 9780521880688


    Args:
        | y : array_like, shape (N,) the values of the time history of the signal.
        | window_size : int the length of the window. Must be an odd integer number.
        | order : int the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        | deriv: int the order of the derivative to compute (default = 0 means only smoothing)


    Returns:
        | ys : ndarray, shape (N) the smoothed signal (or it's n-th derivative).

     Raises:
        | Exception raised for window size errors.
   """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def get_fhwm(wl,tau,normaliseMax=False):
    """Given spectral domain and range, determine full-width half-max domain width

    Returns the FWHM, and the two 50% wavelengths
    """
    # get FWHM https://stackoverflow.com/questions/53445337/implementation-of-a-threshold-detection-function-in-python
    if normaliseMax:
        tau = tau / np.max(tau)
    mask = np.diff(1 * (tau > 0.5) != 0)
    wlcr = np.vstack((wl[:-1][mask],wl[1:][mask]))
    spcr = np.vstack((tau[:-1][mask],tau[1:][mask]))
    lamh = np.zeros((2,))
    # interpolate to get 0.5 crossing
    for i in [0,1]:
        lamh[i] = wlcr[0,i]+(wlcr[1,i]-wlcr[0,i])*(0.5-spcr[0,i])/(spcr[1,i]-spcr[0,i])
    fwhm = lamh[1] - lamh[0]
    return np.abs(fwhm),lamh[0], lamh[1]


def modtran_resample(inspectral, samplingresolution=1,  inwinwidth=1,  outwinwidth=Modtran_Sampling_Resolution,  windowtype=np.bartlett):
    """ Convolve (non-circular) a spectral variable with a window function,
    given the input resolution and input and output window widths.

    This function is normally used on wavenumber-domain spectral data.  The spectral
    data is assumed sampled at samplingresolution wavenumber intervals.
    The inwinwidth and outwinwidth window function widths are full width half-max (FWHM)
    for the window functions for the inspectral and returned spectral variables, respectively.
    The Bartlett function is used as default, but the user can use a different function.
    The Bartlett function is a triangular function reaching zero at the ends. Window function
    width is correct for Bartlett and only approximate for other window functions.

    Spectral convolution is best done in frequency domain ([cm-1] units) because
    the filter or emission line shapes have better symmetry in frequency domain than
    in wavelength domain.

    The input spectral vector must be in spectral density units of cm-1.

    Args:
        | inspectral (np.array[N,] or [N,1]):  spectral variable input  vector (e.g., radiance or transmittance).
        | samplingresolution (float): wavenumber interval between inspectral samples
        | inwinwidth (float): FWHM window width used to obtain the input spectral vector (e.g., spectroradiometer window width)
        | outwinwidth (float): FWHM window width of the output spectral vector after convolution
        | windowtype (function): name of a  numpy/scipy function for the window function

    Returns:
        | outspectral (np.array[N,]):  input vector, filtered to new window width.
        | windowfn (np.array[N,]):  The window function used.

    Raises:
        | No exception is raised.
    """

    winbins = round(2*(outwinwidth/(inwinwidth*samplingresolution)), 0)
    winbins = winbins if winbins%2==1 else winbins+1
    windowfn=windowtype(winbins)
    #np.convolve is unfriendly towards unicode strings
    cmode='same'
    outspectral = np.convolve(windowfn/(samplingresolution*windowfn.sum()),
                        inspectral.reshape(-1, ),mode=cmode)
    return outspectral,  windowfn


def modtran_convolve(spectral_wv, spectral_res, resolution=1, window=Modtran_Sampling_Resolution):
    ''' convolve data in 15 cm^-1 resolution(um)
    '''
    wn, res = convert_spectral_density(spectral_wv, spectral_res, 'ln')
    convolve_res, _ = modtran_resample(res[:, 0], 1, resolution, window) #0.6
    spectral_wv, spectral_res = convert_spectral_density(wn, convolve_res, 'nl')
    return spectral_wv, spectral_res[:, 0]


def modtran_convolve_wn(spectral_wv, spectral_res, resolution=1, window=Modtran_Sampling_Resolution):
    ''' convolve data in 15 cm^-1 resolution(cm^-1)
    '''
    convolve_res, _ = modtran_resample(spectral_res, 1, resolution, window) #0.6
    spectral_wv, spectral_res = convert_spectral_density(spectral_wv, convolve_res, 'nl')
    return spectral_wv, spectral_res[:, 0]


def generate_mask(image: np.ndarray) -> np.ndarray:
    """
    Return a bool array masking 0 and NaN values as False and others as True
    Args:
        image (np.ndarray): Single-band image which is True where we do not want to mask and False where we want to mask.
    """
    zero_mask = image != 0
    nan_mask = image != np.nan
    mask_true = np.logical_and(zero_mask, nan_mask)
    return mask_true


def compute_ndvi(nir: np.ndarray,
                 red: np.ndarray,
                 eps: float = 1e-15,
                 mask=None) -> np.ndarray:
    """Takes the near infrared and red bands of an optical satellite image as input and returns the ndvi: normalized difference vegetation index
    Args:
        nir (np.ndarray): Near-infrared band image
        red (np.ndarray): Red-band image
        eps (float): Epsilon to avoid ZeroDivisionError in numpy
        use_mask (bool): If True, mask NaN and 0 values in input images.
    Returns:
        np.ndarray: Normalized difference vegetation index
    """
    ndvi = (nir - red) / (nir + red + eps)
    ndvi[abs(ndvi) > 1] = np.nan
    if mask is not None:
        ndvi[mask] = np.nan
    return ndvi


def fractional_vegetation_cover(ndvi: np.ndarray) -> np.ndarray:
    """Computes the fractinal vegetation cover matrix
    Args:
        ndvi (np.ndarray):  Normalized difference vegetation index (m x n)
    Returns:
        np.ndarray: Fractional vegetation cover
    """
    if len(ndvi.shape) != 2:
        raise ValueError("NDVI image should be 2-dimensional")
    return ((ndvi - 0.2) / (0.5 - 0.2))**2


def cavity_effect(
    emissivity_veg: float,
    emissivity_soil: float,
    fractional_vegetation_cover: np.ndarray,
    geometrical_factor: float = 0.55,
) -> np.ndarray:
    """Compute the cavity effect matrix
    Args:
        emissivity_veg (float): value of vegetation emissivity
        emissivity_soil (float): value of soil emissivity
        fractional_vegetation_cover (np.ndarray): Fractional vegetation cover image
        geometrical_factor (float, optional): Geometric factor. Defaults to 0.55.
    Returns:
        np.ndarray: Cavity effect numpy array
    """
    to_return = ((1 - emissivity_soil) * emissivity_veg * geometrical_factor *
                 (1 - fractional_vegetation_cover))
    return to_return


def rescale_band(image: np.ndarray,
                 mult: float = 2e-05,
                 add: float = 0.1) -> np.ndarray:
    """rescales the image band
    Args:
        image (np.ndarray): Band 1 - 9, or non Thermal IR bands of the satellite image.
        mult (float, optional): Multiplicative factor. Defaults to 2e-05.
        add (float, optional): Additive factor. Defaults to 0.1.
    Returns:
        np.ndarray: rescaled image of same size as input
    """
    return (mult * image) + 0.1