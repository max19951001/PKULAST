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

from __future__ import unicode_literals
import sys
import os
import logging
import requests
import urllib
import datetime
import numpy as np
from datetime import timezone
import matplotlib.pyplot as plt
from pysolar.solar import get_altitude, get_azimuth
from pkulast.config import DEM_DIR

sys.setrecursionlimit(1500)
LOG = logging.getLogger(__name__)


def get_elevation(lat, lon, method="linear"):
    # “linear”, “nearest”, “zero”, “slinear”, “quadratic”, “cubic”
    import rioxarray
    ds = rioxarray.open_rasterio(DEM_DIR)
    elev = np.squeeze(ds.interp(x=lon, y=lat, method=method).values)
    ds = None
    return elev / 1e3  #km


def get_elevation_online(lat, lon):
    """ get elevation from fixed point.
	"""
    url = r'https://api.opentopodata.org/v1/mapzen?'
    params = {
     'locations': f'{lat}, {lon}',
     'interpolation': 'cubic'
    }
    result = requests.get((url + urllib.parse.urlencode(params)))
    elevation = result.json()['results'][0]['elevation']
    return float(elevation) / 1000

def get_sun_position(lat, lon, date):
    """ solar zenith angle, solar azimuth angle
	"""
    return 90 -  get_altitude(lat, lon, date), get_azimuth(lat, lon, date)

def is_day(lat, lon, date):
    """ check whether specific date somewhere is day or not
	"""
    sza = 90 -  get_altitude(lat, lon, date)
    return sza <=90 and sza >=0

def get_bj_time():
    """ get current beijing time.
	"""
    utc_dc = datetime.datetime.now().replace(tzinfo=timezone.utc)
    bj_dt = utc_dc.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    return bj_dt.strftime('%Y-%m-%d %H:%M:%S')

def sort_data(x_vals, y_vals):
    """Sort the data so that x is monotonically increasing and contains no duplicates."""
    # Sort data
    # (This is needed in particular for EOS-Terra responses, as there are duplicates)
    idxs = np.argsort(x_vals)
    x_vals = x_vals[idxs]
    y_vals = y_vals[idxs]

    # De-duplicate data
    mask = np.r_[True, (np.diff(x_vals) > 0)]
    if not mask.all():
        numof_duplicates = np.repeat(mask, np.equal(mask, False)).shape[0]
        LOG.debug("Number of duplicates in the response function: %d - removing them",
                  numof_duplicates)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]

    return x_vals, y_vals

def convert2str(value):
    """Convert a value to string.
    Args:
        value: Either a str, bytes or 1-element numpy array
    """
    value = bytes2string(value)
    return np2str(value)

def np2str(value):
    """Convert an `numpy.string_` to str.
    Args:
        value (ndarray): scalar or 1-element numpy array to convert
    Raises:
        ValueError: if value is array larger than 1-element or it is not of
                    type `numpy.string_` or it is not a numpy array
    """
    if isinstance(value, str):
        return value

    if hasattr(value, 'dtype') and \
            issubclass(value.dtype.type, (np.str_, np.string_, np.object_)) \
            and value.size == 1:
        value = value.item()
        # python 3 - was scalar numpy array of bytes
        # otherwise python 2 - scalar numpy array of 'str'
        if not isinstance(value, str):
            return value.decode()
        return value

    raise ValueError("Array is not a string type or is larger than 1")

def bytes2string(var):
    """Decode a bytes variable and return a string."""
    if isinstance(var, bytes):
        return var.decode('utf-8')
    return var

def get_cmap(n, name='hsv'):
    ''' Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# from array to bytes
tobytes = lambda array: array.tobytes()
frombytes = lambda array, src: array.frombytes(src)

# read/open file function
def readline(fin): return fin.readline()
def open_file(filename): return open(filename, encoding='iso-8859-1')

# find file
def find_file_path(filename):
    '''
    Search cwd and SPECTRAL_DATA directories for the given file.
    '''
    pathname = None
    dirs = [os.curdir]
    if 'SPECTRAL_DATA' in os.environ:
        dirs += os.environ['SPECTRAL_DATA'].split(os.pathsep)
    for d in dirs:
        testpath = os.path.join(d, filename)
        if os.path.isfile(testpath):
            pathname = testpath
            break
    if not pathname:
        msg = 'Unable to locate file "%s". If the file exists, ' \
          'use its full path or place its directory in the ' \
          'SPECTRAL_DATA environment variable.'  % filename
        raise FileNotFoundError(msg)
    return pathname


import requests
import urllib
import numpy as np
from datetime import datetime
import pylab as plt
from pysolar import solar


__all__ = [
    'str2fp',
    'str2int',
    'str2date',
    'get_elevation',
    'get_sun_position',
    'is_daytime',
    'get_bj_time',
    'get_cmap'
]


def str2fp(num):
    """
    Convert string to float.
    Args:
        num (str): String.
    Returns:
        float: Float.
    """
    try:
        return float(num.strip())
    except Exception as ex:
        raise ValueError(
            f"{ex}, input string can not converted to float point number")


def str2int(num):
    """
    Convert string to int.
    Args:
        num (str): String.
    Returns:
        int: Int.
    """
    try:
        return int(num.strip())
    except Exception as ex:
        raise ValueError(f"{ex},input string can not converted to integer")


def str2date(date_str):
    """
    Convert string to date.
    Args:
        date_str (str): String.
    Returns:
        datetime.datetime: Date.
    """
    year = 1900 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:])
    if day:
        return datetime(year, month, day)
    else:  # dd is not defined
        return datetime(year, month, day + 1)
        # return f"{year}-{month:02d}-na"


def get_elevation(lat, lon):
    """
    Get elevation from latitude and longitude.

    Args:
        lat (float or ndarray): Latitude.
        lon (float or ndarray): Longitude.

    Returns:
        float or ndarray: Elevation. unit:km

    Examples:
        >>> get_elevation(30, 120)
        0.0
    """
    url = r'https://api.opentopodata.org/v1/mapzen?'
    params = {
        'locations': f'{lat}, {lon}',
        'interpolation': 'cubic'
    }
    result = requests.get((url + urllib.parse.urlencode(params)))
    elevation = result.json()['results'][0]['elevation']
    return float(elevation) / 1000


def get_sun_position(lat, lon, date):
    """
    Get sun position from latitude, longitude and date.

    Args:
        lat (float or ndarray): Latitude.
        lon (float or ndarray): Longitude.
        date (datetime.datetime or ndarray): Date.

    Returns:
        float or ndarray: Sun position.

    Examples:
        >>> get_sun_position(30, 120, datetime.datetime(2020, 1, 1))
        0.0
    """
    return 90 - solar.get_altitude(lat, lon, date), solar.get_azimuth(lat, lon, date)


def is_daytime(lat, lon, date):
    """
    Check if it is daytime.

    Args:
        lat (float or ndarray): Latitude.
        lon (float or ndarray): Longitude.
        date (datetime.datetime or ndarray): Date.

    Returns:
        bool or ndarray: True if it is daytime.

    Examples:
        >>> is_daytime(30, 120, datetime.datetime(2020, 1, 1))
        True
    """
    sun_position = get_sun_position(lat, lon, date)
    return sun_position[0] > 0 and sun_position[0] <= 90

def get_bj_time():
    """
    Get Beijing time.

    Returns:
        datetime.datetime: Beijing time.

    Examples:
        >>> get_bj_time()
        datetime.datetime(2020, 1, 1, 0, 0)
    """
    bj_dt = datetime.utcnow() + datetime.timedelta(hours=8)
    return bj_dt.strftime('%Y-%m-%d %H:%M:%S')


def get_cmap(n, name='hsv'):
    """
    Get color map.

    Args:
        n (int): Number of colors.
        name (str): Color map name.

    Returns:
        matplotlib.colors.ListedColormap: Color map.

    Examples:
        >>> get_cmap(10)
        <matplotlib.colors.ListedColormap object at 0x7f0e7f8c8e10>
    """
    return plt.cm.get_cmap(name, n)

def sort_array(x_vals, y_vals):
    """
    Sort array by x_vals in descending order.

    Args:
        x_vals (ndarray): X values.
        y_vals (ndarray): Y values.

    Returns:
        ndarray: Sorted x_vals.
        ndarray: Sorted y_vals.

    Examples:
        >>> sort_array(np.array([1, 2, 3]), np.array([4, 5, 6]))
        (array([1, 2, 3]), array([4, 5, 6]))
    """
    x_vals, y_vals = np.array(x_vals), np.array(y_vals)
    idx = np.argsort(x_vals)
    x_vals = x_vals[idx]
    y_vals = y_vals[idx]

    # Delete duplicate x_vals
    mask = np.r_[True, (np.diff(x_vals)>0)]
    # if not mask.all():
    #     num_of_duplicates = np.repeat(mask, np.equal(mask, False)).shape[0]
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    return x_vals, y_vals
