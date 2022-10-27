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
"""
This module provides configurations for pkulast executions.
Working path for relative spectral response(rsr) files, radiative 
transfer model(rtm), spectral libraries and atmosphere profiles.

"""
import os
import cdsapi
import logging
from datetime import datetime, timedelta


class generic_downloader:
    def __init__(self, url, path):
        self.url = url
        self.path = path
        self.download()

    def download(self):
        pass


def download_era5_reanalysis(lat, lon, acq_time, era5_dir):
    """Download ERA5 Reanalysis Data.
    Args:
        lat (float): Latitude of the profile. unit, deg.
        lon (float): Longitude of the profile. unit, deg.
        acq_time (datetime.datetime): Acquisition utc time of the profile.
    Returns:
        None
    """
    time = []
    filename = '{:.3f}_{:.3f}_'.format(lat, lon) + datetime.strftime(
        acq_time, '%Y%m%d_%H%M')
    single_filename, pressure_filename = era5_dir + filename + '_single.grib', era5_dir + filename + '_pressure.grib'
    if os.path.exists(single_filename) and os.path.exists(pressure_filename):
        return single_filename, pressure_filename
    # time config
    hours = acq_time.hour + acq_time.minute / 60 + acq_time.second / 3600
    delta = hours - int(hours)
    t1 = acq_time - timedelta(hours=delta)
    t2 = t1 + timedelta(hours=1)
    for t in [t1, t2]:
        time.append(datetime.strftime(t, '%H:%M'))
    base_query_args = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'year': str(acq_time.year),
        'month': str(acq_time.month),
        'day': str(acq_time.day),
        'time': time,
    }
    # area config
    if lat is not None and lon is not None:
        area = [lat + 1, lon - 1, lat - 1, lon + 1]
        base_query_args.update({'area': area})
    single_query_args = {
        'variable': [
            'skin_temperature',
            'surface_pressure',
        ],
    }
    single_query_args.update(base_query_args)
    pressure_query_args = {
        'variable': [
            'geopotential',
            'ozone_mass_mixing_ratio',
            'relative_humidity',
            'temperature',
        ],
        'pressure_level': [
            '1',
            '2',
            '3',
            '5',
            '7',
            '10',
            '20',
            '30',
            '50',
            '70',
            '100',
            '125',
            '150',
            '175',
            '200',
            '225',
            '250',
            '300',
            '350',
            '400',
            '450',
            '500',
            '550',
            '600',
            '650',
            '700',
            '750',
            '775',
            '800',
            '825',
            '850',
            '875',
            '900',
            '925',
            '950',
            '975',
            '1000',
        ],
    }
    pressure_query_args.update(base_query_args)
    c = cdsapi.Client(quiet=True)
    c.retrieve('reanalysis-era5-single-levels', single_query_args,
               single_filename)
    c.retrieve('reanalysis-era5-pressure-levels', pressure_query_args,
               pressure_filename)
    logging.info(f'{filename}_single.grib and {filename}_pressure.grib were downloaded!')
    return single_filename, pressure_filename