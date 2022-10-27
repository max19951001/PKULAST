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

''' Thermal Utilities
'''
import logging
import numpy as np
import dask as da
from pkulast.constants import *

LOG = logging.getLogger(__name__)

def planckian(wl, T, unit='um'):
    ''' planck function.
    Get radiance under specific wavelength and temperature.

    Args:
        wl: wavelength(unit:micron) list/numpy.array object
        T: temperature float

    Returns:
        array of radiance under specific wavelength and temperature
    '''
    if unit == 'nm':
        wl = np.array(wl) / 1e3
    wl =  np.asarray(wl)
    C1 = 2 * H_PLANCK * C_SPEED * C_SPEED
    C2 = H_PLANCK * C_SPEED / (K_BOLTZMANN * T)
    with np.errstate(over='ignore'):
        ret = C1 / (np.power(wl / 1E6, 4) * wl) / (np.exp(C2 / (wl / 1E6))- 1.0)
    return ret

def inverse_planckian(wl, radiance_array):
    ''' inverse of planck function.
    Get temperature from radiance under specific wavelength.

    Args:
        wl: wavelength(unit:micron) list/numpy.array object
        radiance_array: array of radiance corresponding to wl.

    Returns:
        array of temperature from radiance array under specific wavelength.
    '''
    wl =  np.asarray(wl)
    radiance_array = np.asarray(radiance_array)
    C1 = 2 * H_PLANCK * C_SPEED * C_SPEED
    C2 = H_PLANCK * C_SPEED / K_BOLTZMANN
    return C2 / (wl / 1E6) / np.log(C1 / np.power(wl / 1E6,4) / wl /radiance_array + 1.0)

def planck_wl(wl, T, unit='um'):
    if unit == 'nm':
        wl = np.array(wl) / 1e9
    elif unit == 'um':
        wl = np.array(wl) / 1e6
    else:
        raise ValueError(f'unsupported units: {unit}, use mircometer or nanometer instead')
    return planck(wl, T) / 1e6

def planck(wave, temperature, wavelength=True):
    """Derive the Planck radiation as a function of wavelength or wavenumber.

    SI units.
    _planck(wave, temperature, wavelength=True)
    wave = Wavelength/wavenumber or a sequence of wavelengths/wavenumbers (m or m^-1)
    temp = Temperature (scalar) or a sequence of temperatures (K)


    Output: Wavelength space: The spectral radiance per meter (not micron!)
            Unit = W/m^2 sr^-1 m^-1

            Wavenumber space: The spectral radiance in Watts per square meter
            per steradian per m-1:
            Unit = W/m^2 sr^-1 (m^-1)^-1 = W/m sr^-1

            Converting from SI units to mW/m^2 sr^-1 (cm^-1)^-1:
            1.0 W/m^2 sr^-1 (m^-1)^-1 = 1.0e5 mW/m^2 sr^-1 (cm^-1)^-1

    """
    units = ['wavelengths', 'wavenumbers']
    if wavelength:
        LOG.debug("Using {0} when calculating the Blackbody radiance".format(
            units[(wavelength is True) - 1]))

    if np.isscalar(temperature):
        temperature = np.array([temperature, ], dtype='float64')
    elif isinstance(temperature, (list, tuple)):
        temperature = np.array(temperature, dtype='float64')

    shape = temperature.shape
    if np.isscalar(wave):
        wln = np.array([wave, ], dtype='float64')
    else:
        wln = np.array(wave, dtype='float64')

    if wavelength:
        const = 2 * H_PLANCK * C_SPEED ** 2
        nom = const / wln ** 5
        arg1 = H_PLANCK * C_SPEED / (K_BOLTZMANN * wln)
    else:
        nom = 2 * H_PLANCK * (C_SPEED ** 2) * (wln ** 3)
        arg1 = H_PLANCK * C_SPEED * wln / K_BOLTZMANN

    with np.errstate(divide='ignore', invalid='ignore'):
        # use dask functions when needed
        np_ = np if isinstance(temperature, np.ndarray) else da
        arg2 = np_.where(np_.greater(np.abs(temperature), EPSILON),
                         np_.divide(1., temperature), np.nan).reshape(-1, 1)

    if isinstance(arg2, np.ndarray):
        # don't compute min/max if we have dask arrays
        LOG.debug("Max and min - arg1: %s  %s",
                  str(np.nanmax(arg1)), str(np.nanmin(arg1)))
        LOG.debug("Max and min - arg2: %s  %s",
                  str(np.nanmax(arg2)), str(np.nanmin(arg2)))

    try:
        exp_arg = np.multiply(arg1.astype('float64'), arg2.astype('float64'))
    except MemoryError:
        LOG.warning(("Dimensions used in numpy.multiply probably reached "
                     "limit!\n"
                     "Make sure the Radiance<->Tb table has been created "
                     "and try running again"))
        raise

    if isinstance(exp_arg, np.ndarray) and exp_arg.min() < 0:
        LOG.debug("Max and min before exp: %s  %s",
                  str(exp_arg.max()), str(exp_arg.min()))
        LOG.warning("Something is fishy: \n" +
                    "\tDenominator might be zero or negative in radiance derivation:")
        dubious = np.where(exp_arg < 0)[0]
        LOG.warning(
            "Number of items having dubious values: " + str(dubious.shape[0]))

    with np.errstate(over='ignore'):
        denom = np.exp(exp_arg) - 1
        rad = nom / denom
        radshape = rad.shape
        if wln.shape[0] == 1:
            if temperature.shape[0] == 1:
                return rad[0, 0]
            else:
                return rad[:, 0].reshape(shape)
        else:
            if temperature.shape[0] == 1:
                return rad[0, :]
            else:
                if len(shape) == 1:
                    return rad.reshape((shape[0], radshape[1]))
                else:
                    return rad.reshape((shape[0], shape[1], radshape[1]))

def convert_wave_unit(w):
    ''' convert wavelength(nm) to wavenumber(cm^-1) or vice versa
  '''
    return 1E7 / w
