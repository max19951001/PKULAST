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

"""Physical utilities."""


from pkulast import constants

def vmr2mixing_ratio_for_ozone(x):
    r"""Convert volume mixing ratio to mass mixing ratio.

    .. math::
        w = \frac{x}{1 - x} \frac{M_w}{M_d}

    Parameters:
        x (float or ndarray): Volume mixing ratio.

    Returns:
        float or ndarray: Mass mixing ratio.

    Examples:
        >>> vmr2mixing_ratio(0.04)
        0.025915747437955664
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_ozone

    return x / (1 - x) * Mw / Md


def pressure2altitude(pressures, t, p):
    """
    Convert pressure to altitude.

    Parameters:
        pressures (float or ndarray): Pressure(s) in Pa.
        t (float or ndarray): Temperature in K.
        p (float or ndarray): Pressure in Pa.

    Returns:
        float or ndarray: Altitude(s) in m.

    Examples:
        >>> pressure2altitude(101325, 288.15, 101325)
        0.0
    """
    return (p / pressures - 1) * (t / 288.15) * constants.g0

def geopotential_to_height(geopotential):
    """
    Convert geopotential to height.

    Parameters:
        geopotential (float or ndarray): Geopotential in m**2/s**2.

    Returns:
        float or ndarray: Height in km.

    Examples:
        >>> geopotential_to_height(0)
        0.0
    """
    return constants.earth_radius * geopotential / (constants.earth_radius -
                                                    geopotential) / 1000
