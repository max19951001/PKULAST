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
Atmospheric profile.
====================

Extracts atmospheric profile from a variety of NWP products.

"""
import re
import os
import json
import shutil
import random
import requests
import calendar
import tarfile
import numpy as np
import xarray as xr
from glob import glob
import subprocess
from time import sleep
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pkulast.config import *
from pkulast.constants import *
from metpy.calc import mixing_ratio_from_relative_humidity, relative_humidity_from_mixing_ratio
from metpy.units import units
from pkulast.utils.physics.atmosphere import vmr2mixing_ratio_for_ozone, relative_humidity2vmr, integrate_water_vapor
from pkulast.utils.physics import thermodynamics
from pkulast.utils.atmosphere import *


NWPConfiguration = get_nwp_config()
PROFILE_TYPES = list(NWPConfiguration.keys())

class NWPLibrary(object):
    proxy = None
    def __init__(self, product='GFS', proxy=None):
        if product not in PROFILE_TYPES:
            raise ValueError(
             f'atmosphere profile {product} is not include in the libray!')
        self._profile_type = product
        self._reader = self._get_reader(proxy)

    @property
    def profile_type(self):
        return self._profile_type

    @profile_type.setter
    def profile_type(self, value):
        if value not in PROFILE_TYPES:
            raise ValueError(
             f'atmosphere profile {value} is not include in the libray!')
        self._profile_type = value

    @property
    def available_profile_types(self):
        return PROFILE_TYPES

    def extract(self, acq_time, lat, lon, method='linear'):  #linear
        ''' Extract atmpospheric profile for specific time/location using nearest interpolation method
        '''
        if method not in INTERPOLATION_METHODS:
            raise ValueError(
             f'Interpolation method {method} is not included, options are {INTERPOLATION_METHODS}'
            )
        if not isinstance(acq_time, datetime):
            raise ValueError(
             'input parameter acq_time should be a datetime object')
        return self._reader.extract_atm_profile(acq_time, lat, lon, method=method)

    def _get_reader(self, proxy):
        if self._profile_type == 'GFS':
            return GFSHandler(proxy)
        elif self._profile_type == 'MODIS':
            return None
        elif self._profile_type == 'GDAS':
            return GDASHandler(proxy)
        elif self._profile_type == 'GDAS25':
            return GDAS25Handler(proxy)
        elif self._profile_type == 'ERA5':
            return ERA5Handler(proxy)
        elif self._profile_type == 'MERRA2':
            return MERRA2Handler(proxy)
        elif self._profile_type == 'JRA55':
            return JRA55Handler(proxy)
        elif self._profile_type == 'CFSv2':
            return CFSv2Handler(proxy)
        elif self._profile_type == 'DOE':
            return DOEHandler(proxy)
        elif self._profile_type == 'ERA-Interim':
            return None


class GDASLibrary:
    def __init__(self, ncepfilepath=GDAS_DIR, proxy=None):
        self.ncepfilepath = ncepfilepath
        self.profile_name = "GDAS"
        self.proxy = proxy
        self.execuable = None
        self.configuration = NWPConfiguration["GDAS"]
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
                f"directory {ncepfilepath} is not a valid directory!")

    def extract(self, acq_time, lat_0, lon_0, method="linear"):  # linear
        """Extract atmpospheric profile for specific time/location using nearest interpolation method"""
        if method not in INTERPOLATION_METHODS:
            raise ValueError(
                f"Interpolation method {method} is not included, options are {INTERPOLATION_METHODS}"
            )
        if not isinstance(acq_time, datetime):
            raise ValueError(
                "input parameter acq_time should be a datetime object")
        ncepfiles, self.delta = self._check_ncep_folder(acq_time)
        self.layer_number = 0  # layer count
        self.acq_time = acq_time
        self.lat = lat_0
        self.lon = (lon_0 + 360) % 360
        # lat_min = math.copysign(1, lat) * int(
        #     abs(lat) / self.configuration['resolution_lat']
        # ) * self.configuration['resolution_lat']
        # lat_max = lat_min + math.copysign(
        #     1, lat) * self.configuration['resolution_lat']
        lat_min = (int(self.lat / self.configuration["resolution_lat"]) *
                   self.configuration["resolution_lat"])
        lat_max = lat_min + self.configuration["resolution_lat"]

        lat_nearest = (lat_min
                       if abs(self.lat - lat_min) < abs(self.lat - lat_max)
                       else lat_max)

        lon_min = (int(self.lon / self.configuration["resolution_lon"]) *
                   self.configuration["resolution_lon"])
        lon_max = lon_min + self.configuration["resolution_lon"]
        lon_nearest = (lon_min
                       if abs(self.lon - lon_min) < abs(self.lon - lon_max)
                       else lon_max)

        self.locs = (list(product(
            (lat_min, lat_max),
            (lon_min, lon_max))) if method != "nearest" else
                     ((lat_nearest, lon_nearest), ))
        TAPE5_CARD2C = []
        P_SURFACE = []
        T_SURFACE = []
        TCWV_all = []
        P_all = []
        H_all = []
        TMP_all = []
        RH_all = []
        O3_all = []
        import rasterio
        for grb2filename in ncepfiles:
            grb2_path = os.path.join(self.ncepfilepath, grb2filename)
            cmd_str = f'{self.execuable} "{grb2_path}" -s'
            lines = (subprocess.check_output(
                cmd_str, shell=False).decode("utf-8").strip().split("\n"))
            with rasterio.open(grb2_path) as src:
                surface_pressure_indices = self.match(":MSLET:mean",
                                                      lines)  # PRMSL MSLET
                if not surface_pressure_indices:
                    surface_pressure_indices = self.match(":PRMSL:", lines)
                PWAT_indices = self.match(":PWAT:", lines)
                skin_temperature_indices = self.match(":TMP:surface", lines)
                if not skin_temperature_indices:
                    skin_temperature_indices = self.match(":TMP:sfc:", lines)
                HGT_indices, HGT_levels = self.match(r":HGT:\d+(\.\d+)? mb:",
                                                     lines,
                                                     return_level=True)
                TMP_indices, TMP_levels = self.match(r":TMP:\d+(\.\d+)? mb:",
                                                     lines,
                                                     return_level=True)
                O3MR_indices, O3MR_levels = self.match(r":O3MR:\d+(\.\d+)? mb:",
                                                       lines,
                                                       return_level=True)
                RH_indices, RH_levels = self.match(r":RH:\d+(\.\d+)? mb:",
                                                   lines,
                                                   return_level=True)
                for lat, lon in self.locs:
                    surface_pressure = (list(
                        src.sample([(lon, lat)],
                                   indexes=surface_pressure_indices))[0][0] /
                                        100)  # hPa
                    skin_temperature = list(
                        src.sample(
                            [(lon, lat)],
                            indexes=skin_temperature_indices))[0][0]  # K
                    if skin_temperature < 100:
                        skin_temperature = skin_temperature + 273.15
                    TCWV = list(src.sample([(lon, lat)],
                                           indexes=PWAT_indices))[0][0] / 10
                    # convert kg/m^2 to g/cm^2
                    HGT = list(src.sample([(lon, lat)],
                                          indexes=HGT_indices))[0]  # km
                    O3MR = list(src.sample([
                        (lon, lat)
                    ], indexes=O3MR_indices))[0] * 1e3  # convert g/g to g/kg
                    TMP = list(src.sample([(lon, lat)],
                                          indexes=TMP_indices))[0]  # K
                    if min(TMP) < 100:
                        TMP = TMP + 273.15
                    RH = list(src.sample([(lon, lat)],
                                         indexes=RH_indices))[0]  # %
                    RH[RH == 0] = 1e-16
                    sorted_indices = np.argsort(TMP_levels)[::-1]
                    pressure_levels = TMP_levels
                    heights = np.array(list(map(h2z, HGT)))
                    RH = np.array(
                        list(
                            fix_null(dict(zip(RH_levels, RH)), pressure_levels,
                                     heights).values()))
                    O3 = np.array(
                        list(
                            fix_null(dict(zip(O3MR_levels, O3MR)),
                                     pressure_levels, heights).values()))
                    # P_SURFACE.append(Pb)
                    T_SURFACE.append(skin_temperature)
                    TCWV_all.append(TCWV)
                    P_all.append(pressure_levels[sorted_indices])
                    H_all.append(heights[sorted_indices])
                    TMP_all.append(TMP[sorted_indices])
                    RH_all.append(RH[sorted_indices])
                    O3_all.append(O3[sorted_indices])
        # spatial interpolation
        if method == "linear":
            T_SURFACE = [
                spatial_interpolation(self.locs, values, (self.lat, self.lon))
                for values in self._split_array(T_SURFACE)
            ]
            TCWV_all = [
                spatial_interpolation(self.locs, values, (self.lat, self.lon))
                for values in self._split_array(TCWV_all)
            ]
            P_all = self._spatial_interpolate(P_all)
            H_all = self._spatial_interpolate(H_all)
            TMP_all = self._spatial_interpolate(TMP_all)
            RH_all = self._spatial_interpolate(RH_all)
            O3_all = self._spatial_interpolate(O3_all)

        # temporal interpolation
        Ts = temporal_interpolation([0, 6], T_SURFACE, self.delta)
        TCWV = temporal_interpolation([0, 6], TCWV_all, self.delta)
        P = list(
            map(
                lambda x: temporal_interpolation([0, 6], x, self.delta),
                np.transpose(P_all),
            ))
        H = list(
            map(
                lambda x: temporal_interpolation([0, 6], x, self.delta),
                np.transpose(H_all),
            ))
        TMP = list(
            map(
                lambda x: temporal_interpolation([0, 6], x, self.delta),
                np.transpose(TMP_all),
            ))
        RH = list(
            map(
                lambda x: temporal_interpolation([0, 6], x, self.delta),
                np.transpose(RH_all),
            ))
        O3 = list(
            map(
                lambda x: temporal_interpolation([0, 6], x, self.delta),
                np.transpose(O3_all),
            ))
        layer_number = len(P)

        LOG.debug(
            f"Lon = {lon_0:>10.6f} Lat = {lat_0:>10.6f} Name = PyIRT_Produced Ts = {Ts:>10.3f}\n"
        )
        LOG.debug(
            f'Date_Time = {acq_time.strftime("%Y/%m/%d, %H:%M:%S")} TCWV = {TCWV:>.3f} g/cm^2\n'
        )
        LOG.debug(
            "Height(km)  Pressure(hPa)     AirTemp(K)          RH(%)    Ozone(g/kg)\n"
        )
        Card2C = f"   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n"
        for lyr in range(layer_number - 1, -1, -1):
            LOG.debug(
                f"{H[lyr]:>10.4f}{P[lyr]:>15.4f}{TMP[lyr]:>15.4f}{RH[lyr]:>15.4f}{O3[lyr]:>15.8f}\n"
            )
            Card2C += f"{H[lyr]:>10.3f}{P[lyr]:>10.3f}{TMP[lyr]:>10.3f}{RH[lyr]:>10.3e}{0:>10.3e}{O3[lyr]:>10.3e}{atmosphere_model(abs(lat_0), acq_time.month):>10s}\n"
        LOG.debug("\n")
        params = np.c_[H, P, TMP, RH, O3]
        profile = Profile(
            self.profile_name,
            acq_time,
            lat_0,
            lon_0,
            Ts,
            layer_number,
            params,
            Card2C,
            TCWV,
        )
        return profile  # patch_profile(profile)

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration["resolution_t"]
        tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
                hours=t * self.configuration["resolution_t"] - delta)
            if datetime(current.year, current.month, current.day,
                        current.hour) > datetime(2007, 12, 6, 6):
                ext = "grib2"
                self.execuable = WGRIB2_EXE
            else:
                ext = "grib1"
                self.execuable = WGRIB_EXE
            file = f"{ext}/{current.year}/{current.year}.{current.month:0>2d}/fnl_{current.year}{current.month:0>2d}{current.day:0>2d}_{current.hour:0>2d}_00.{ext}"
            filename = os.path.basename(file)
            tiles_file.append(filename)
            if not os.path.exists(f"{self.ncepfilepath}/{filename}"):
                download_ucar_file(
                    self.configuration["url"],
                    file,
                    f"{self.ncepfilepath}/{filename}",
                    self.proxy,
                )
        return tiles_file, delta

    def _spatial_interpolate(self, data):
        return [
            list(
                map(
                    lambda x: spatial_interpolation(self.locs, x,
                                                    (self.lat, self.lon)),
                    np.transpose(values),
                )) for values in self._split_array(data)
        ]

    def _split_array(self, data):
        return [data[0:4], data[4:]]

    def match(self, match_str, lines, return_level=False):
        indices = []
        levels = []
        for i, line in enumerate(lines):
            if re.search(match_str, line):
                res = re.search(
                    r"\d+(\.\d+)?",
                    re.search(match_str, line).group().split(":")[2])
                if res:
                    levels.append(float(res.group()))
                indices.append(i + 1)
        return indices if not return_level else (indices,
                                                 np.array(levels, dtype=float))


class TIGRLibrary(object):
    ''' TIGR Atmospheric Profile Library 
    '''
    def __init__(self):
        self.count = TIGR_TOTALCOUNT
        self.profiles = []
        self.pressures = PRESSURE_N_LEVELS
        self._load()

    def _load(self):
        self.date_time = []
        self.temperature_profile = []
        self.mixing_ratio_profile = []
        self.ozone_profile = []
        self.latitude = []
        self.longitude = []
        self.surface_pressure = []
        self.skin_temperature = []
        self.tcwv = []
        with open(TIGR_FILENAME, 'r') as fn:
            self.lines = fn.readlines()
            for index in range(self.count):
                line = self.lines[index * TOTAL_LINE:(index + 1) * TOTAL_LINE]
                no, lon, lat, date = list(
                 map(str2int, line[FIRST_LINE].strip().split()))
                profile_type = f'TIGR {no}'
                lon = lon / 100
                lat = lat / 100
                date_time = str2date(str(date))
                layer_number = NLEVEL
                TMP = list(
                 map(str2fp,
                  ' '.join(line[SECOND_LINE:THIRD_LINE]).split())) # K
                Ts, Ps = list(map(str2fp, line[THIRD_LINE].split()))
                H2O = list(
                 map(str2fp,
                  ' '.join(line[FOURTH_LINE:FIFTH_LINE]).split())) # g/g
                # RH = mmr2rh(H2O, TMP, PRESSURE_N_LEVELS) # abandon typhon
                RH = relative_humidity_from_mixing_ratio(
                    PRESSURE_N_LEVELS * units.hPa, TMP * units.kelvin,
                    H2O * units.dimensionless) * 100

                OZONE = np.array(
                 list(
                  map(str2fp, ' '.join(
                   line[FIFTH_LINE:TOTAL_LINE]).split()))) * 1e3 # convert g/g to g/kg
                # HEIGHT_N_LEVELS = pressure_to_altitude(PRESSURE_N_LEVELS, Ts, Ps)
                HEIGHT_N_LEVELS = get_height(PRESSURE_N_LEVELS, TMP, RH)
                tcwv = get_tcwv(H2O, PRESSURE_N_LEVELS)
                # p = list(reversed(PRESSURE_N_LEVELS))
                # x = list(reversed(H2O))
                # tcwv = vibeta(p, x, p[0]) * 10 / 9.80665
                params = np.c_[list(reversed(HEIGHT_N_LEVELS)),
                   list(reversed(PRESSURE_N_LEVELS)),
                   list(reversed(TMP)),
                   list(reversed(RH)),
                   list(reversed(OZONE))]
                self.profiles.append(
                 Profile(profile_type,
                   date_time,
                   lat,
                   lon,
                   Ts,
                   layer_number,
                   profile=params,
                   TCWV=tcwv))
                self.date_time.append(date_time)
                self.temperature_profile.append(TMP)
                self.mixing_ratio_profile.append(H2O)
                self.ozone_profile.append(OZONE)
                self.latitude.append(lat)
                self.longitude.append(lon)
                self.surface_pressure.append(Ps)
                self.skin_temperature.append(Ts)
                self.tcwv.append(tcwv)

    def __str__(self):
        return "Thermodynamic Initial Guess Retrieval(TIGR) dataset"

    def __repr__(self):
        pass

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.profiles[index]

class SeeborLibrary(object):
    ''' Seebor Atmospheric Profile Library
    '''
    def __init__(self):
        f = open(Seebor_FILENAME, 'rb')
        data = np.fromfile(file=f, dtype=np.float32)
        f.close()
        self.pressures = Seebor_PRESSURE_LEVEL
        data = data.reshape((15704, 338))

        self.temperature_profile, self.mixing_ratio_profile, self.ozone_profile, self.latitude, self.longitude, self.surface_pressure, self.skin_temperature, self.wind_speed, self.tpw, self.ecosystem, self.elevation, self.fraction_land, self.year, self.month, self.day, self.hour, self.profile_type, self.frequency, self.emissivity_spectra = (
         data[:, range(0, 101)], data[:, range(101, 202)],
         data[:, range(202, 303)], data[:, 303], data[:, 304], data[:, 305],
         data[:, 306], data[:, 307], data[:, 308], data[:, 309], data[:,
          310],
         data[:, 311], data[:, 312], data[:, 313], data[:, 314], data[:,
          315],
         data[:, 316], data[:, range(317, 327)], data[:, range(327, 337)])
        self.count = Seebor_TOTALCOUNT
        self.layer_number = Seebor_NLEVEL

    def get_surface_pressure(self, index):
        return self.surface_pressure[index]

    def get_skin_temperature(self, index):
        return self.skin_temperature[index]

    def get_wind_speed(self, index):
        return self.wind_speed[index]

    def get_tcwv(self, index):
        return self.tpw[index]

    def get_ecosystem(self, index):
        return self.ecosystem[index]

    def get_elevation(self, index):
        return self.elevation[index]

    def get_fraction_land(self, index):
        return self.fraction_land[index]

    def get_profile_type(self, index):
        return self.profile_type[index]

    def get_frequency(self, index):
        return self.frequency[index]

    def get_emissivity_spectra(self, index):
        return self.emissivity_spectra[index]

    def get_mixing_ratio_profile(self, index):
        return self.mixing_ratio_profile[index]

    def get_temperature_profile(self, index):
        return self.temperature_profile[index]

    def get_ozone_profile(self, index):
        return self.ozone_profile[index]

    def get_latitude(self, index):
        return self.latitude[index]

    def get_longitude(self, index):
        return self.longitude[index]

    def get_time(self, index):
        return datetime(int(self.year[index]), int(self.month[index]), int(self.day[index]), int(self.hour[index]))

    def get_profile(self, index):
        TMP = self.temperature_profile[index]
        H2O = self.mixing_ratio_profile[index]
        OZONE = vmr2mixing_ratio_for_ozone(self.ozone_profile[index] * 1e-6) * 1e3 # convert ppmv to g/g then to g/kg
        lat = self.latitude[index]
        lon = self.longitude[index]
        Ps = self.surface_pressure[index]
        Ts = self.skin_temperature[index]
        profile_type = self.profile_type[index]
        emissivity_spectra = self.emissivity_spectra[index]
        year = self.year[index]
        month = self.month[index]
        day = self.day[index]
        hour = self.hour[index]
        date_time = datetime(int(year), int(month), int(day), int(hour))
        tcwv = get_tcwv(H2O, Seebor_PRESSURE_LEVEL)
        # RH = mmr2rh(H2O, TMP, Seebor_PRESSURE_LEVEL) # abandon typhon
        RH = relative_humidity_from_mixing_ratio(
            Seebor_PRESSURE_LEVEL * units.hPa, TMP * units.kelvin,
            H2O * units.dimensionless) * 100

        HEIGHT_N_LEVELS = get_height(Seebor_PRESSURE_LEVEL, TMP, RH)
        params = np.c_[list(reversed(HEIGHT_N_LEVELS)),
           list(reversed(Seebor_PRESSURE_LEVEL)),
           list(reversed(TMP)),
           list(reversed(RH)),
           list(reversed(OZONE))]
        return Profile(profile_type,
           date_time,
           lat,
           lon,
           Ts,
           self.layer_number,
           profile=params,
           TCWV=tcwv)

    def __str__(self):
        return "Seebor profile dataset"

    def __repr__(self):
        pass

    def __len__(self):
        return self.count

    def __iter__(self):
        self.current_index = -1
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.count:
            raise StopIteration
        return self.get_profile(self.current_index)

    def __getitem__(self, index):
        return self.get_profile(index)

class StandardLibrary(object):
    """ standard profiles
    """

    @classmethod
    def patch_profile(cls, p):
        profile_type = p.profile_type
        layer_number = p.layer_number
        acq_time = p.acq_time
        # self.profile = profile  # H, P, TMP, RH, O3
        lat = p.latitude
        lon = p.longitude
        Ts = p.Ts
        H = p.H
        P = p.P
        TMP = p.TMP
        RH = p.RH
        O3 = p.O3
        s = StandardLibrary()
        if lat <= 25:
            ref_p = s['tropical']
        elif (lat <= 65):
            if acq_time.month >= 4 and acq_time.month < 10:
                ref_p = s['midlatitude summer']
            else:
                ref_p = s['midlatitude winter']
        else:
            if acq_time.month >= 4 and acq_time.month < 10:
                ref_p = s['subarctic summer']
            else:
                ref_p = s['subarctqic winter']
        highest_height = H[-1]
        for i, h in enumerate(ref_p.H):
            if h > highest_height + 5 and h <= 100:
                layer_number += 1
                H = np.append(H, h)
                P = np.append(P, ref_p.P[i]) if ref_p.P[i] > 1e-16 else np.append(P, 1e-16)
                TMP = np.append(TMP, ref_p.TMP[i])
                RH = np.append(RH, ref_p.RH[i])
                O3 = np.append(O3, ref_p.O3[i])
        params = np.c_[H, P, TMP, RH, O3]
        return Profile(profile_type,
                       acq_time,
                       lat,
                       lon,
                       Ts,
                       layer_number,
                       profile=params)

    def __init__(self):
        self.profiles = defaultdict()
        for k, v in ICRCCM_STANDARD_ATMOSPHERES.items():
            filename = ICRCCM_STANDARD_ATMOSPHERES_DIR + v
            self.profiles[k] = self._load_profile(k, filename)
            # Z = dat[:, 1]
            # P = dat[:, 2]
            # T = dat[:, 3]
            # H2O = dat[:, 4]
            # O3 = dat[:, 5]
            # CO2 = dat[:, 6]
            # CO = dat[:, 7]
            # CH4 = dat[:, 8]
            # N2O = dat[:, 9]
            # O2 = dat[:, 10]
            # NH3 = dat[:, 11]
            # NO = dat[:, 12]
            # NO2 = dat[:, 13]
            # SO2 = dat[:, 14]
            # HNO3 = dat[:, 15]
            #  * 0.4462 * 18 * 1e-4
            # relative_humidity2vmr
            # print(k, integrate_water_vapor(H2O * 0.4462 * 18 * 1e-4, P, T, Z * 1e3))
            # print(integrate_column(H2O, Z))
            # print(k, np.trapz(H2O, Z))
            # print(k, vibeta(Z, H2O, Z[-1], Z[-1], Z[0]) * 10 / 9.80665)
    def keys(self):
        return list(self.profiles.keys())

    def get_profile(self, k):
        return self.profiles[k]

    def iter_profiles(self):
        return list(self.profiles.items())

    def _load_profile(self, k, filename):
        data = defaultdict(list)
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.strip()[0] == '!':
                    continue
                elif '!' in line:
                    layer_number = int(line.split('!')[0])
                elif '*END' in line:
                    break
                elif '*' in line:
                    name = line.split(' ')[0].replace('*', '')
                else:
                    for val in line.split(','):
                        if val.strip():
                            data[name].append(float(val))
        date_time = datetime(1980, 1, 1, 12, 0, 0)
        h = np.array(data['HGT']) # km
        p = np.array(data['PRE']) # hPa
        t = np.array(data['TEM']) # K
        h2o = np.array(data['H2O']) * 1e-6 # convert ppmv to dimensionless
        co2 = data['CO2'] # ppmv
        o3 = np.array(data['O3']) * 1e-6 * O3_MASS * 1e3 / MOL_MASS # convert ppmv to g/kg
        n2o = data['N2O'] # ppmv
        co = data['CO'] # ppmv
        ch4 = data['CH4'] # ppmv
        o2 = data['O2'] #ppmv
        # rh = vmr2relative_humidity(h2o, p * 1e2, t) * 1e2, abandon typhon utils
        h2o_mmr = h2o * H2O_MASS / MOL_MASS
        rh = relative_humidity_from_mixing_ratio(
            p * units.hPa, t * units.kelvin,
            h2o_mmr * units.dimensionless) * 100
        params = np.c_[h, p, t, rh, o3]
        return Profile(k,
                       date_time,
                       0,
                       0,
                       t[0],
                       layer_number,
                       profile=params)
    def __getitem__(self, k):
        return self.get_profile(k)

    def __str__(self):
        return "Standard Library dataset"

class Profile(object):
    """ Atmospheric profile
    """
    @classmethod
    def parse_array(cls, profile_type, acq_time, lat, lon, array):
        Ts = array[0]
        TCWV = array[1]
        VIS = array[2]
        layer_count = int((len(array) - 3) / 5)
        H = array[3:3 + layer_count]
        P = array[3 + layer_count:3 + 2 * layer_count]
        TMP = array[3 + 2 * layer_count:3 + 3 * layer_count]
        RH = array[3 + 3 * layer_count:3 + 4 * layer_count]
        O3 = array[3 + 4 * layer_count:]
        params = np.c_[H, P, TMP, RH, O3]
        return Profile(profile_type,
           acq_time,
           lat,
           lon,
           Ts,
           layer_number=layer_count,
           profile=params,
           TCWV=TCWV,
           VIS=VIS)

    def __init__(self,
     profile_type,
     acq_time,
     lat,
     lon,
     Ts,
     layer_number=0,
     profile=None,
     card2c=None,
     TCWV=-999,
     VIS=-999):
        self.profile_type = profile_type
        self.layer_number = layer_number
        self.acq_time = acq_time
        self.latitude = lat
        self.longitude = lon
        self.Ts = Ts
        self.H = None
        self.P = None
        self.TMP = None
        self.RH = None
        self.O3 = None
        if profile is not None:
            self.H, self.P, self.TMP, self.RH, self.O3 = profile.transpose()
        if TCWV == -999:
            vmr = relative_humidity2vmr(self.RH * 1e-2, self.P * 1e2, self.TMP)
            TCWV = integrate_water_vapor(vmr, self.P * 1e2, self.TMP, self.H * 1e3) / 10
        self.MMR = mixing_ratio_from_relative_humidity(self.P * units.hPa, self.TMP * units.kelvin, self.RH * units.percent)
        self._card2c = card2c
        self.TCWV = TCWV
        self.VIS = VIS
        self.Ta = self.get_effective_temperature()

    @property
    def profile(self):
        return np.c_[self.H, self.P, self.TMP, self.RH, self.O3]

    def save(self, save_path):
        ''' Generate atmosphere profile'''
        with open(save_path, "w") as fn:
            fn.write(
             f'Lon = {self.longitude:>10.6f} Lat = {self.latitude:>10.6f} Name = {self.profile_type} Ts = {self.Ts:>10.3f}\n'
            )
            fn.write(
             f'DateTime = {self.acq_time.strftime("%Y/%m/%d, %H:%M:%S")} TCWV = {self.TCWV:.3f} g/cm^2 VIS = {self.VIS:.3f} km\n'
            )
            fn.write(
             'Height(km)  Pressure(hPa)     AirTemp(K)          RH(%)    Ozone(g/kg)\n'
            )
            for lyr in self.profile:
                fn.write(
                 f'{lyr[0]:>10.4f}{lyr[1]:>15.4f}{lyr[2]:>15.4f}{lyr[3]:>15.4f}{lyr[4]:>15.8f}\n'
                )
            fn.write("\n")

    def get_effective_temperature(self):
        """ Get effective mean atmospheric temperature
        """
        vmr = relative_humidity2vmr(self.RH * 1e-2, self.P * 1e2, self.TMP)
        mmr = thermodynamics.vmr2mixing_ratio(vmr)
        num=np.trapz( np.array(self.TMP).reshape(-1, 1) * np.array(mmr).reshape(-1, 1), self.P, axis=0)[0]
        den=np.trapz(np.array(mmr).reshape(-1, 1), self.P, axis=0)[0]
        return num/den

    def plot(self):
        ''' plot profile
        '''
        config = {
            "font.family": "serif",
            "font.size": 16,
            "mathtext.fontset": "stix",
            "font.family": ["Times New Roman"],
            "font.serif": ["SimSun"],
        }
        plt.rcParams.update(config)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
        fig.suptitle(
         f'{self.profile_type} profile; lon:{self.longitude:>8.3f},lat:{self.latitude:>8.3f},datetime: {self.acq_time.strftime("%Y/%m/%d, %H:%M:%S")} TCWV={self.TCWV:.3f}(g/cm^2)'
        )
        axs[0, 0].plot(self.P, self.H)
        # axs[0, 0].set_title('Height(km)-Pressure(hPa)')
        axs[0, 0].set(xlabel='Pressure(hPa)', ylabel='Height(km)')

        axs[0, 1].plot(self.TMP, self.H, 'tab:orange')
        # axs[0, 1].set_title('Height(km)-Temperature(K)')
        axs[0, 1].set(xlabel='Temperature(K)', ylabel='Height(km)')

        axs[1, 0].plot(self.RH, self.H, 'tab:green')
        # axs[1, 0].set_title('Height(km)-Relative Humidity(%)')
        axs[1, 0].set(xlabel='Relative Humidity(%)', ylabel='Height(km)')

        axs[1, 1].plot(self.O3, self.H, 'tab:red')
        # axs[1, 1].set_title('Height(km)-Ozone(g/kg)')
        axs[1, 1].set(xlabel='Ozone(g/kg)', ylabel='Height(km)')
        plt.show()
        plt.close()

    def savefig(self, save_path, dpi=300):
        ''' plot profile
        '''
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
         f'{self.profile_type} profile; lon:{self.longitude:>8.3f},lat:{self.latitude:>8.3f},datetime: {self.acq_time.strftime("%Y/%m/%d, %H:%M:%S")} '
        )
        axs[0, 0].plot(self.P, self.H)
        # axs[0, 0].set_title('Height(km)-Pressure(hPa)')
        axs[0, 0].set(xlabel='Pressure(hPa)', ylabel='Height(km)')

        axs[0, 1].plot(self.TMP, self.H, 'tab:orange')
        # axs[0, 1].set_title('Height(km)-Temperature(K)')
        axs[0, 1].set(xlabel='Temperature(K)', ylabel='Height(km)')

        axs[1, 0].plot(self.RH, self.H, 'tab:green')
        # axs[1, 0].set_title('Height(km)-Relative Humidity(%)')
        axs[1, 0].set(xlabel='Relative Humidity(%)', ylabel='Height(km)')

        axs[1, 1].plot(self.O3, self.H, 'tab:red')
        # axs[1, 1].set_title('Height(km)-Ozone(g/kg)')
        axs[1, 1].set(xlabel='Ozone(g/kg)', ylabel='Height(km)')
        plt.savefig(save_path, dpi=dpi)
        plt.close()

    def get_card2c(self):
        '''generate MODTRAN tape5 CARD2C'''
        # if not self._card2c:
        self._card2c = f'{self.layer_number:>5d}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in self.profile:
            self._card2c += f"{lyr[0]:>10.3f}{lyr[1]:>10.3e}{lyr[2]:>10.3e}{lyr[3]:>10.3e}{0:>10.3e}{lyr[4]:>10.3e}{atmosphere_model(self.latitude, self.acq_time.month):>10s}\n"
        return self._card2c

    def compute_tcwv(self):
        """compute total column water vapor.
        """
        vza=0
        mult=False
        is_satellite=True
        if is_satellite:
            ITYPE = 3
        else:
            ITYPE = 2
        IEMSCT = 0
        if mult:
            if is_satellite:
                IMULT = -1
            else:
                IMULT = 1
        else:
            IMULT = 0
        card_1 = f'TMF 7{ITYPE:5d}{IEMSCT:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
        card_1a = 'FFF  8 0.0   400.000  0.000000          01 F F T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'
        current_ground_altitude = self.H[0]
        card_2 = f'    1    0    0    0    0    0    23.000     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
        card_2c = self.get_card2c()
        H1ALT = 100
        H2ALT = 0 if ITYPE == 3 else self.H[0]
        card_3 = f'{H1ALT:>10.3f}{H2ALT:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'
        card_3a1 = ''
        card_3a2 = ''
        card_4 = '   400.000 33333.000       1.0       2.0RM              F  1             !card4\n'
        card_5 = '    0 !card5'
        tp5 = card_1 + card_1a + card_2 + \
            card_2c + card_3 + card_3a1 + card_3a2  + card_4+ card_5
        with open(RTM_MODTRAN_DIR + "/mod5root.in", "w") as mod5in:
            mod5in.write(f"{TRANS_DIR}.tp5")
        with open(f'{TRANS_DIR}.tp5', "w") as tranf:
            tranf.write(tp5)
        cwd = os.getcwd()
        os.chdir(RTM_MODTRAN_DIR)
        output = ''
        try:
            output = subprocess.check_output('run_mod5.bat', shell=False).decode()
        except Exception as e:
            # print(e)
            pass
        if 'Error' in output:
            os.chdir(cwd)
            raise Exception(output.split('********************************************************')[-1].strip())
        while not os.path.exists(TRANS_DIR + '.tp6'):
            continue
        os.chdir(cwd)
        res = 0
        with open(TRANS_DIR + '.tp6') as f:
            for line in f.readlines():
                if 'FINAL' in line and 'GM / CM2' in line:
                    res = float(line.split()[1])
        return res

    def get_tcwv_typhon(self):
        vmr = relative_humidity2vmr(self.RH * 1e-2, self.P * 1e2, self.TMP)
        TCWV = integrate_water_vapor(vmr, self.P * 1e2) / 10 #, self.TMP, self.H * 1e3
        return TCWV

    def get_tcwv_alpha(self):
        vmr = relative_humidity2vmr(self.RH * 1e-2, self.P * 1e2, self.TMP)
        mmr = thermodynamics.vmr2mixing_ratio(vmr)
        TCWV = get_tcwv(mmr, self.P)
        return TCWV

    def get_tcwv_beta(self):
        vmr = relative_humidity2vmr(self.RH * 1e-2, self.P * 1e2, self.TMP)
        mmr = thermodynamics.vmr2mixing_ratio(vmr)
        TCWV = get_tcwv_beta(mmr, self.P)
        return TCWV

    def flatten(self):
        ret = []
        ret.append(self.Ts)
        ret.append(self.TCWV)
        ret.append(self.VIS)
        ret.extend(self.H.flatten())
        ret.extend(self.P.flatten())
        ret.extend(self.TMP.flatten())
        ret.extend(self.RH.flatten())
        ret.extend(self.O3.flatten())
        return np.array(ret, dtype=np.float64)

    def scale_wv(self, amp):
        scale_factor = random.gauss(1, amp)
        self.RH *= scale_factor
        self.RH[self.RH > 100] = 100

    def scale_t(self, amp):
        scale_amp = random.gauss(0, amp)
        self.TMP += scale_amp

    def scale_o3(self, amp):
        scale_factor = random.gauss(1, amp)
        self.O3 *= scale_factor

    def __repr__(self):
        return f"Atmosphere Profile: Longitude:{self.longitude:>5}, Latitude:{self.latitude:>5}, Datetime: {self.acq_time.strftime('%Y/%m/%d, %H:%M:%S')}\n"

    def __str__(self):
        return self.get_card2c()

class CLAR(object):
    def __init__(self):
        pass


def ClearSkyTIGR946(selected_type=('tropical', 'mid-latitude', 'polar'), selected_tcwv=[0, 10], ref_indices=False):
    """
    type:
        tropical
        mid-latitude
        polar
    """
    layer_number = 40
    profiles = []
    indices = []
    count = -1
    with open(ClearSkyTIGR946_FILE, 'r') as af:
        row_index = 0
        acq_time = None
        lat = 0
        lon = 0
        Ts = 0
        TCWV = 0
        profile_type = ''
        ap = []
        for line in af.readlines():
            row_index += 1
            if row_index == 1:
                meta = line.split(':')
                profile_type = meta[2].split()[0]
                lon = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', meta[3])[0])
                lon = -lon if 'W' in meta[3] else lon
                lat = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', meta[4])[0])
                lat = -lat if 'S' in meta[4] else lat
                acq_time = datetime(int(meta[5][:4]), int(meta[5][5:7]), 1)
                Ts = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', meta[6])[0])
                TCWV = float(re.findall(r'[-+]?[0-9]*\.?[0-9]+', meta[7])[0])
            elif row_index == 2:
                continue
            elif row_index == layer_number + 3:
                params = np.asarray(ap)
                count += 1
                if TCWV >= selected_tcwv[0] and TCWV <= selected_tcwv[1]:
                    if profile_type in selected_type:
                        profiles.append(
                        Profile(profile_type,
                        acq_time,
                        lat,
                        lon,
                        Ts,
                        layer_number,
                        params,
                        TCWV=TCWV))
                        indices.append(count)
                ap = []
                row_index = 0
            else:
                ap.append([float(i) for i in line.strip().split()])
        return (profiles, indices) if ref_indices else profiles

def parse_profile(atmfile):
    ''' n * (Height(km)  Pressure(hPa)     AirTemp(K)          RH(%)    Ozone(g/kg) * layer_number)
    '''
    meta = ''
    with open(atmfile, "r") as af:
        row_index = 0
        data = []
        ap = []
        for line in af.readlines():
            row_index += 1
            if row_index < 4:
                meta += line
                continue
            elif not line.strip():
                data.append(np.asarray(ap))
                ap = []
                row_index = 0
                break
            else:
                ap.append([float(i) for i in line.strip().split()])
    info = meta.split()
    lon = float(info[2])
    lat = float(info[5])
    profile_type = info[8]
    ts = float(info[11])
    dt = info[14] + info[15]
    acq_time = datetime.strptime(dt, '%Y/%m/%d,%H:%M:%S')
    tcwv =  float(info[18])
    vis =  float(info[22])
    profile_header = [ts, tcwv, vis]
    profile_header.extend(data[0].flatten(order='F'))
    return Profile.parse_array(profile_type, acq_time, lat, lon, profile_header)

def extract_profile(acq_time, lat, lon, method='linear'):
    nc = NWPCube(acq_time, (lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5))
    return nc.interp(lat, lon, spatial_method=method)

def patch_profile(p):
    return StandardLibrary.patch_profile(p)


class NWPCube(object):
    """ (lat x lon x time) 3D cube
    """
    proxy = None
    
    def __init__(self,
     acq_time,
     extent,
     nceppath=GDAS_DIR,
     product='GDAS',
     tile_count=1):
        if not isinstance(acq_time, datetime):
            raise ValueError(
             f"parameter {acq_time} must be a datetime.datetime object")
        self.acq_time = acq_time
        self.nceppath = nceppath
        self.extent = extent
        self.configuration = NWPConfiguration[product]
        self.product = product
        self.tile_count = tile_count
        self.tiles = []
        self._check_ncep_folder()
        self.set_extent(self.extent)

    def set_extent(self, extent, stride=1):
        for tile in self.tiles_file:
            g = Grib(f"{self.nceppath}/{tile}")
            profiles = g.extract(extent=extent, stride=stride)
            lats, lons = g.latlons(extent=extent, stride=stride)
            nt = NWPTile(profiles, lats, lons, g.acq_time, extent)
            self.tiles.append(nt)

    def locate(self, lat, lon, method='linear'):
        """ spatial interp options: 'linear', 'nearest', 'cubic'
        """
        return [tile.interp(lat, lon, method) for tile in self.tiles]

    def interp(self,
      lat,
      lon,
      acq_time=None,
      spatial_method='linear',
      temporal_method='linear'):
        """ temporal interp options: linear, nearest, zero, slinear, quadratic, cubic
        """
        from scipy.interpolate import interp1d
        acq_time = self.acq_time if acq_time is None else acq_time
        times = [calendar.timegm(d.timetuple()) for d in self.corrd_t]
        values = [p.flatten() for p in self.locate(lat, lon, spatial_method)]
        ret = interp1d(times, values, axis=0, kind=temporal_method)([
         calendar.timegm(acq_time.timetuple()),
        ])
        return Profile.parse_array(self.product, acq_time, lat, lon,
           np.squeeze(ret))

    def _check_ncep_folder(self):
        acq_time = self.acq_time
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        self.corrd_t = []
        self.tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
                hours=t * self.configuration['resolution_t'] - delta)
            if datetime(current.year, current.month, current.day,
                        current.hour) > datetime(2007, 12, 6, 6):
                ext = 'grib2'
            else:
                ext = 'grib1'
            file = f"{ext}/{current.year}/{current.year}.{current.month:0>2d}/fnl_{current.year}{current.month:0>2d}{current.day:0>2d}_{current.hour:0>2d}_00.{ext}"
            filename = os.path.basename(file)
            self.corrd_t.append(current)
            self.tiles_file.append(filename)
            if not os.path.exists(f"{self.nceppath}/{filename}"):
                download_ucar_file(self.configuration['url'], file,
                                   f"{self.nceppath}/{filename}", self.proxy)

    # def _check_ncep_folder(self):
    #     hours = self.acq_time.hour + self.acq_time.minute / 60.0 + self.acq_time.second / 3600.0
    #     delta = hours % self.configuration['resolution_t']
    #     self.corrd_t = []
    #     self.tiles_file = []
    #     # threads = []
    #     for t in range(2 * self.tile_count):
    #         current = self.acq_time + timedelta(
    #          hours=(t + 1 - self.tile_count) *
    #          self.configuration['resolution_t'] - delta)
    #         filename = f"gfs_4_{current.year}{current.month:0>2d}{current.day:0>2d}_{current.hour:0>2d}00_000.grb2"
    #         self.corrd_t.append(current)
    #         self.tiles_file.append(filename)
    #         if not os.path.exists(f"{self.nceppath}/{filename}"):
    #             download_gfs_grib(current)
    #             # threads.append(Thread(target=download_gfs_grib, args=(current, )))
    #     # for thread in threads:
    #     # 	thread.start()
    #     # for thread in threads:
    #     # 	thread.join()

class NWPTile(object):
    """ profiles tile
    """
    def __init__(self, profiles, lats, lons, acq_time, extent):
        self.profiles = profiles
        self.shape = profiles.shape
        self.profile_type = profiles[0, 0].profile_type
        self.lats = lats
        self.lons = lons
        self.acq_time = acq_time
        self.extent = extent
        self._lat_lons = list(zip(self.lats.flatten(), self.lons.flatten()))
        self._values = [p.flatten() for p in self.profiles.flatten()]

    def interp(self, lat, lon, method='linear'):
        """ interp: {'linear', 'nearest', 'cubic'}
        """
        from scipy.interpolate import griddata
        ret = griddata(self._lat_lons,
           self._values, [
         (lat, lon),
           ],
           method=method)
        return Profile.parse_array(self.profile_type, self.acq_time, lat, lon,
           np.squeeze(ret))

class Grib(object):
    """ parse grib1/2 files and extract profiles
    """
    def __init__(self, ncepfilename):
        self.filename = ncepfilename
        self.datasets = defaultdict()
        if not os.path.exists(os.path.abspath(ncepfilename)):
            raise ValueError(f"file {ncepfilename} does not exist!")
        self.profile_type = os.path.basename(self.filename)[:3]
        self._load_datasets()

    def available_datasets(self):
        if self.datasets is None:
            return None
        return self.datasets.keys()

    def _load_datasets(self):
        import pygrib
        pg = pygrib.open(self.filename)
        # single-level
        try:
            self.datasets['Surface temperature'] = self._loads(
             pg(name='Temperature', typeOfLevel='surface'))
            self.datasets['Surface pressure'] = self._loads(
             pg.select(name='Surface pressure'))
            self.datasets['Geopotential Height'] = self._loads(
             pg(name='Geopotential Height', typeOfLevel='isobaricInhPa'))
            self.datasets['Temperature'] = self._loads(
             pg(name='Temperature', typeOfLevel='isobaricInhPa'))
            self.datasets['Relative humidity'] = self._loads(
             pg(name='Relative humidity', typeOfLevel='isobaricInhPa'))
            self.datasets['Ozone mixing ratio'] = self._loads(
             pg(name='Ozone mixing ratio', typeOfLevel='isobaricInhPa'))
            self.datasets['Precipitable water'] = self._loads(
             pg(name='Precipitable water'))
            self.layer_count = len(self.datasets['Temperature']['object'])
            self.pressures = self.datasets['Temperature']['levels']
            self.acq_time = self.datasets['Temperature']['object'][0].validDate
            self.datasets['Visibility'] = self._loads(
             pg.select(name='Visibility', typeOfLevel='surface'))
        except Exception as ex:
            print(f'Grib {ex}')

    def plot(self, dataset, level=0, extent=None, proj='mill'):
        if extent is None:
            extent = (-90, 90, 0, 360)
        plt.figure(figsize=[12, 10])
        values, lats, lons = self.datasets[dataset]['object'][level].data(
         *extent)
        llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = lons.min(), lats.min(
        ), lons.max(), lats.max()
        from mpl_toolkits.basemap import Basemap
        m = Basemap(llcrnrlon=llcrnrlon,
           llcrnrlat=llcrnrlat,
           urcrnrlon=urcrnrlon,
           urcrnrlat=urcrnrlat,
           projection=proj,
           area_thresh=10000,
           resolution='l')
        x, y = m(lons, lats)
        m.pcolormesh(x, y, values, cmap='Spectral_r',
         shading='auto')  # , vmax=.05
        m.drawcounties(zorder=1, color="black")
        m.drawcoastlines(color="black", linewidth=2)
        m.drawstates()
        m.drawrivers(linewidth=0.5, color="blue")
        m.drawmapboundary()
        m.drawparallels(np.linspace(llcrnrlat, urcrnrlat, 10),
         labels=[1, 1, 0, 0])
        m.drawmeridians(np.linspace(llcrnrlon, urcrnrlon, 10),
         labels=[0, 0, 0, 1])
        plt.colorbar(label=self.datasets[dataset]['units'])
        plt.show()

    def subset(self, dataset, extent=None):
        if dataset not in self.datasets.keys():
            raise ValueError(f"{dataset} not in datasets")
        if extent is None:
            extent = (-90, 90, 0, 360)
        n_levels = len(self.datasets[dataset]['object'])
        lats, lons = self.latlons(extent)
        ret = np.zeros([n_levels, lats.shape[0], lats.shape[1]])
        for level in range(n_levels):
            values, _, _ = self.datasets[dataset]['object'][level].data(
             *extent)
            ret[level, :, :] = values
        return np.squeeze(ret)

    def latlons(self, extent, stride=1):
        _, LATs, LONs = self.datasets['Temperature']['object'][0].data(*extent)
        row, col = int(LATs.shape[0] / stride), int(LATs.shape[1] / stride)
        lats = np.fromfunction(lambda i, j: LATs[i * stride, j * stride],
          (row, col),
          dtype=int)
        lons = np.fromfunction(lambda i, j: LONs[i * stride, j * stride],
          (row, col),
          dtype=int)
        return lats, lons

    def levels(self, dataset):
        return self.datasets[dataset]['levels']

    def extract(self, extent, stride=1):
        """  extract profiles
        """
        LATs, LONs = self.latlons(extent)
        row, col = int(LATs.shape[0] / stride), int(LATs.shape[1] / stride)
        Ts, Ps, HGT, TMP, RH, OZONE, PWAT, VIS = list(
         map(lambda dataset: self.subset(dataset, extent),
          self.datasets.keys()))

        def fix_nodata(i, j):
            r, c = i * stride, j * stride
            lat = LATs[r, c]
            lon = LONs[r, c]
            vis = VIS[r, c]
            pwat = PWAT[r, c]
            ts = Ts[r, c]
            hgt = HGT[:, r, c]
            h = list(map(h2z, hgt))
            tmp = TMP[:, r, c]

            rh = list(
             fix_null(
              dict(
               zip(self.datasets['Relative humidity']['levels'],
             RH[:, r, c])), self.pressures, h).values())
            o3 = list(
             fix_null(
              dict(
               zip(self.datasets['Ozone mixing ratio']['levels'],
             OZONE[:, r, c] * 1000)), self.pressures,
              h).values())
            card2c = f'   {self.layer_count}    0    0                           0.0    0     1.000    28.964  !card2c\n'
            for lyr in range(self.layer_count):
                card2c += f"{h[lyr]:>10.3f}{self.pressures[lyr]:>10.3f}{tmp[lyr]:>10.3f}{rh[lyr]:>10.3e}{0:>10.3e}{o3[lyr]:>10.3e}{atmosphere_model(lat, self.acq_time.month):>10s}\n"
            params = np.c_[h, self.pressures, tmp, rh, o3]
            return Profile(self.profile_type, self.acq_time, lat, lon, ts,
               self.layer_count, params, card2c, pwat, vis)

        profiles = np.fromfunction(np.vectorize(fix_nodata),
           shape=(row, col),
           dtype=int)
        return profiles

    def locate(self, lat, lon):
        extent = (lat, lat, lon, lon)
        Ts, Ps, HGT, TMP, RH, OZONE, PWAT, VIS = list(
         map(lambda dataset: self.subset(dataset, extent),
          self.datasets.keys()))
        h = list(map(h2z, HGT))
        rh = list(
         fix_null(
          dict(zip(self.datasets['Relative humidity']['levels'], RH)),
          self.pressures, h).values())
        o3 = list(
         fix_null(
          dict(
           zip(self.datasets['Ozone mixing ratio']['levels'],
         OZONE * 1000)), self.pressures, h).values())
        card2c = f'   {self.layer_count}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(self.layer_count):
            card2c += f"{h[lyr]:>10.3f}{self.pressures[lyr]:>10.3f}{TMP[lyr]:>10.3f}{rh[lyr]:>10.3e}{0:>10.3e}{o3[lyr]:>10.3e}{atmosphere_model(lat, self.acq_time.month):>10s}\n"
        params = np.c_[h, self.pressures, TMP, rh, o3]
        return Profile(self.profile_type, self.acq_time, lat, lon, Ts,
           self.layer_count, params, card2c, PWAT, VIS)

    def _loads(self, gribs):
        n_levels = len(gribs)
        levels = np.array([grb_element['level'] for grb_element in gribs])
        indexes = np.argsort(levels)[::-1]  # highest pressure first
        objects = []
        for i in range(n_levels):
            objects.append(gribs[indexes[i]])
        dicts = {
         'object': objects,
         'units': objects[0]['units'],
         'levels': levels[indexes]
        }
        return dicts

def download_gfs_grib(current):
    """ download gfs achieve grb2 files
    """
    y_m = f"{current.year}{current.month:0>2d}"
    y_m_d = f"{y_m}{current.day:0>2d}"
    filename = f"gfs_4_{y_m_d}_{current.hour:0>2d}00_000.grb2"
    if os.path.exists(f"{GFS_DIR}/{filename}"):
        return
    print(f'downloading file {filename}...')
    try:
        url = f"{NCEI_GFSANL4_URL}/{y_m}/{y_m_d}/gfs_4_{y_m_d}_{current.hour:0>2d}00_000.grb2"
        cwd = os.getcwd()
        os.chdir(GFS_DIR)
        subprocess.check_call(f'{WGET_EXE} "{url}" -q -P "{GFS_DIR}"', shell=True)  # -q
        os.chdir(cwd)
    except subprocess.CalledProcessError:
        try:
            url = f"{NCEI_GFSANL4_URL_HIS}/{y_m}/{y_m_d}/gfsanl_4_{y_m_d}_{current.hour:0>2d}00_000.grb2"
            cwd = os.getcwd()
            os.chdir(GFS_DIR)
            subprocess.check_call(f'{WGET_EXE} "{url}" -P "{GFS_DIR}" -O {filename}', shell=True)
            os.chdir(cwd)
        except subprocess.CalledProcessError:
            # ncep file is archived, request form required to download
            print(
             f'{filename} is archived, more time to be cost to download it from remote server!'
            )
            af = ArchiveFile()
            af.download(current, current, station=f'{current.hour:0>2d}')

    print(f'{filename} was downloaded!')
    return filename

def download_era5_grib(acq_time, lat=None, lon=None):
    time = []
    filename = '{:.3f}_{:.3f}_'.format(lat, lon) + datetime.strftime(acq_time, '%Y%m%d_%H%M')
    single_filename, pressure_filename = ERA5_DIR + filename  + '_single.grib', ERA5_DIR + filename  + '_pressure.grib'
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
       'skin_temperature','surface_pressure',
      ],
     }
    single_query_args.update(base_query_args)
    pressure_query_args =   {
      'variable': [
       'geopotential', 'ozone_mass_mixing_ratio', 'relative_humidity',
       'temperature',
      ],
      'pressure_level': [
       '1', '2', '3',
       '5', '7', '10',
       '20', '30', '50',
       '70', '100', '125',
       '150', '175', '200',
       '225', '250', '300',
       '350', '400', '450',
       '500', '550', '600',
       '650', '700', '750',
       '775', '800', '825',
       '850', '875', '900',
       '925', '950', '975',
       '1000',
      ],
     }
    pressure_query_args.update(base_query_args)
    import cdsapi
    c = cdsapi.Client(quiet=True)
    c.retrieve(
     'reanalysis-era5-single-levels',
     single_query_args,
     single_filename)
    c.retrieve(
     'reanalysis-era5-pressure-levels',
     pressure_query_args,
     pressure_filename)
    # print(f'{filename}_single.grib and {filename}_pressure.grib were downloaded!')
    return single_filename, pressure_filename

def download_ucar_file(dataset, file, savepath, proxy=None):
    # Authenticate
    dataset = dataset.replace('datasets', 'data')
    ret = requests.post(UCAR_LOGIN_URL, data=UCAR_PARAMS)
    if ret.status_code != 200:
        print('Bad Authentication')
        print(ret.text)
        raise Exception("UCAR authenticate failed!")
    if proxy is None:
        req = requests.get(dataset + file, cookies = ret.cookies, allow_redirects=True, stream=True)
    else:
        proxies = {
         'http': f'http://{proxy}/',
         'https': f'http://{proxy}/'
        }
        req = requests.get(dataset + file, cookies = ret.cookies, allow_redirects=True, stream=True, proxies=proxies)
    filesize = int(req.headers['Content-length'])
    from pkulast.utils.io import check_file_status
    with open(savepath, 'wb') as outfile:
        chunk_size=1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(savepath, filesize)
    check_file_status(savepath, filesize)

def download_ucar_multi_files(dataset, files, savepath, proxy=None):
    # Authenticate
    cmd = f'ncar_downloader.exe --uid {UCAR_PARAMS["email"]} --password {UCAR_PARAMS["passwd"]} --download_dir {savepath} --url '
    dataset = dataset.replace('datasets', 'data')
    for file in files:
        name = os.path.basename(file)
        if os.path.exists(os.path.join(savepath, name)):
            continue
        cmd += ' "{}" '.format(dataset + file)
    if proxy is not None:
        cmd += f' --proxy "{proxy}"'
    subprocess.check_call(cmd, shell=False)

def download_reanalysis2_files(files, savepath):
    url = 'https://ftp.cpc.ncep.noaa.gov/wd51we/reanalysis-2/6hr/pgb/'
    cmd = f'earthdata_downloader.exe --download_dir {savepath} --url '
    # for file in files:
    #     name = os.path.basename(file)
    #     if os.path.exists(os.path.join(savepath, name)):
    #         continue
    #     cmd += ' "{}" '.format(dataset + file)
    # result = subprocess.check_output(cmd, shell=False)

def download_merra2_nc(acq_time, lat=None, lon=None, proxy=None):
    cmd = f'merra2_downloader.exe --download_dir "{MERRA2_DIR}" '
    outfile = os.path.join(MERRA2_DIR, 'MERRA2_{:.3f}_{:.3f}_'.format(lat, lon) + datetime.strftime(acq_time, '%Y%m%d_%H%M') + '.nc')
    if os.path.exists(outfile):
        return outfile
    start_date_str = f' --initial_year {acq_time.year} --initial_month {acq_time.month} --initial_day {acq_time.day}'
    end_date_str = f' --final_year {acq_time.year} --final_month {acq_time.month} --final_day {acq_time.day}'
    geo = f' --bottom_left_lat {lat - 2} --bottom_left_lon {lon - 2} --top_right_lat {lat + 2} --top_right_lon {lon + 2}'
    cmd = f'{cmd} {start_date_str} {end_date_str} {geo}'
    if proxy is not None:
        cmd += f' --proxy "{proxy}"'
    try:
        filename = subprocess.check_output(
         f'{cmd}',
         shell=False)
        if filename.decode('utf-8') == '':
            raise RuntimeError('MERRA2 data download failed!')
        merra_name = [name.strip() for name in filename.decode('utf-8').split('\n') if 'MERRA2' in name][0]
        infile = os.path.join(MERRA2_DIR, merra_name)
        shutil.move(infile, outfile)
        return outfile
    except Exception as e:
        LOG.error(e)
        raise RuntimeError('MERRA2 data download failed!')

class ArchiveFile(object):
    """ NOAA Archive Information Request System (AIRS) Interface.
    https://www.ncdc.noaa.gov/has/HAS.DsSelect
    not available
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(AIRS_HEADERS)
        self._request(AIRS_GFSANL4_URL, parseJson=False, JsonBody=False)

    def download_all(self, start_date, end_date):
        """ download all stations gfsanl files from start_date to end_date
        """
        from threading import Thread
        threads = []
        for station in AIRS_GFSANL4_STATIONS:
            threads.append(
             Thread(target=self.download,
              args=(start_date, end_date, station)))
        LOG.debug(
         "AIRS downloading ncep reanalysis grib2 files from internet, this may cost a few minutes..."
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        LOG.debug("AIIRS downloading tasks are finished!")

    def download(self, start_date, end_date, station):
        """ download single station gfsanl files from start_date to end_date
        """
        info = {
         'stations': station,
         'begyear': start_date.year,
         'begmonth': f'{start_date.month:0>2d}',
         'begday': f'{start_date.day:0>2d}',
         'endyear': end_date.year,
         'endmonth': f'{end_date.month:0>2d}',
         'endday': f'{end_date.day:0>2d}',
        }
        AIRS_GFSANL4_BODY.update(info)
        ret = self._request(AIRS_ORDER_URL,
          body=AIRS_GFSANL4_BODY,
          parseJson=False,
          JsonBody=False)
        if ret.status_code != 200:
            raise Exception(
             'NOAA Archive Information Request System (AIRS) not available!'
            )
        from bs4 import BeautifulSoup
        href = BeautifulSoup(ret.text, features='html.parser').select_one(
         'td[class="val"]').a.attrs['href']
        ret = self._request(AIRS_CHECK_STATUS_URL.format(href),
          parseJson=False,
          JsonBody=False)
        href = BeautifulSoup(ret.text, features='html.parser').select_one(
         'div[class="has-progress-bar"]').a.attrs['href']
        test = requests.get(href)
        while not test.status_code == 200:
            sleep(5)
            test = requests.get(href)
        soup = BeautifulSoup(test.text, features='html.parser')
        filenames = soup.findAll('a', attrs={'href': re.compile("^gfsanl")})
        for obj in filenames:
            filename = obj.text
            url = href + filename
            filepath = GFS_DIR + filename
            if not os.path.exists(filepath):
                self._download(url)
                self._unzip(filepath)
                os.remove(filepath)

    def _request(self,
     url: str,
     body=None,
     parseJson=True,
     JsonBody=True,
     Referer=None):
        if Referer != None:
            self.session.headers.update({"Referer": Referer})
        if body == None:
            ret = self.session.get(url)
        else:
            self.session.headers.update({
             "Content-Type": ("application/json" if JsonBody else
               "application/x-www-form-urlencoded")
            })
            ret = self.session.post(
             url, data=(json.dumps(body) if JsonBody else body))
        if parseJson:
            return json.loads(ret.text)
        else:
            return ret

    def _download(self, url):
        print(
         f'downloading ncep reanalysis grib2 files from {url}, this may cost a few minutes...'
        )
        try:
            command = f'{WGET_EXE} "{url}" -q -P {GFS_DIR}'
            subprocess.check_call(command, shell=True)
        except Exception as e:
            raise ValueError(f"Errors occur: {e}!")

    def _unzip(self, filepath):
        reT = re.compile(r'.*?000.grb2')
        for tar_filename in glob(filepath):
            try:
                t = tarfile.open(tar_filename, 'r')
                t.extractall(
                 GFS_DIR,
                 members=[m for m in t.getmembers() if reT.search(m.name)])
                t.close()
            except IOError as e:
                LOG.info(f'unzip error occur: {e}')

class GFSHandler:
    '''Read NWP Grib2 files
    '''
    def __init__(self, proxy, ncepfilepath=GFS_DIR):
        self.ncepfilepath = ncepfilepath
        self.profile_name = 'GFS'
        self.proxy = proxy
        self.configuration = NWPConfiguration['GFS']
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def extract_atm_profile(self, acq_time, lat_0, lon_0, method='linear'):
        """ extract atmpospheric profiles (nearest, linear)
        """
        ncepfiles, self.delta = self._check_ncep_folder(acq_time)
        self.layer_number = 0  # layer count
        self.acq_time = acq_time
        self.lat = lat_0
        self.lon = (lon_0 + 360) % 360
        # lat_min = math.copysign(1, lat) * int(
        #     abs(lat) / self.configuration['resolution_lat']
        # ) * self.configuration['resolution_lat']
        # lat_max = lat_min + math.copysign(
        #     1, lat) * self.configuration['resolution_lat']
        lat_min = int(
         self.lat / self.configuration['resolution_lat']
        ) * self.configuration['resolution_lat']
        lat_max = lat_min + self.configuration['resolution_lat']


        lat_nearest = lat_min if abs(self.lat - lat_min) < abs(self.lat -
          lat_max) else lat_max

        lon_min = int(self.lon / self.configuration['resolution_lon']
          ) * self.configuration['resolution_lon']
        lon_max = lon_min + self.configuration['resolution_lon']
        lon_nearest = lon_min if abs(self.lon - lon_min) < abs(self.lon -
          lon_max) else lon_max

        self.locs = list(product(
         (lat_min, lat_max),
         (lon_min, lon_max))) if method != 'nearest' else ((lat_nearest,
         lon_nearest), )
        TAPE5_CARD2C = []
        P_SURFACE = []
        T_SURFACE = []
        TCWV_all = []
        P_all = []
        H_all = []
        TMP_all = []
        RH_all = []
        O3_all = []
        for grb2filename in ncepfiles:
            for lat, lon in self.locs:
                TMP = defaultdict()  # air temperature(K)
                RH = defaultdict()  # relative humidity(%)
                O3 = defaultdict()  # Ozone(g/kg)
                HGT = defaultdict()  # Height(km)

                P = []  # Presure(hPa)
                H = []  # Height(km)
                TCWV = 0
                Ts = 0.0  # underlying surface temperature
                Pb_pattern = r'-match ":MSLET:mean"'  # MSLSA MSLET MSLMA PRMSL
                Ts_pattern = r'-match ":TMP:surface"'
                HGT_pattern = r'-match ":HGT{0,1}:[[:digit:]]*[.]{0,1}[[:digit:]]*[[:blank:]]*mb"'
                PWAT_pattern = r'-match ":PWAT"'
                O3_pattern = r'-match ":O3MR{0,1}:[[:digit:]]*[.]{0,1}[[:digit:]]*[[:blank:]]*mb"'
                TMP_pattern = r'-match ":TMP:[[:digit:]]*[.]{0,1}[[:digit:]]*[[:blank:]]*mb"'
                RH_pattern = r'-match ":RH{0,1}:[[:digit:]]*[.]{0,1}[[:digit:]]*[[:blank:]]*mb"'
                cmd = f'{WGRIB2_EXE} "{self.ncepfilepath+grb2filename}" {{}} -s  -lon {lon} {lat}'
                for line in run_command(cmd.format(Pb_pattern)):
                    Pb = float(line.split("val=")[1].strip()) / 100  # hPa
                for line in run_command(cmd.format(Ts_pattern)):
                    Ts = float(line.split("val=")[1].strip())
                for line in run_command(cmd.format(PWAT_pattern)):
                    val = float(line.split("val=")[1].strip())
                    TCWV = val / 10  # convert kg/m^2 to g/cm^2
                for line in run_command(cmd.format(O3_pattern)):
                    O3[float(
                     line.split("mb:anl")[0].split("O3MR:")
                     [1].strip())] = float(line.split(
                      "val=")[1].strip()) * 1000  # convert g/g to g/kg
                for line in run_command(cmd.format(TMP_pattern)):
                    TMP[float(
                     line.split("mb:anl")[0].split("TMP:")
                     [1].strip())] = float(line.split("val=")[1].strip())
                for line in run_command(cmd.format(RH_pattern)):
                    val = float(line.split("val=")[1].strip())
                    RH[float(line.split("mb:anl")[0].split("RH:")
                       [1].strip())] = val if val else 1e-16
                for line in run_command(cmd.format(HGT_pattern)):
                    val = float(line.split("val=")[1].strip())
                    HGT[float(
                     line.split("mb:anl")[0].split("HGT:")
                     [1].strip())] = val

                P = list(TMP.keys())
                H = list(map(h2z, HGT.values()))
                # H = pressure_to_altitude(P, Ts, Pb)
                TMP = list(TMP.values())
                RH = list(fix_null(RH, P, H).values())
                O3 = list(fix_null(O3, P, H).values())
                # P_SURFACE.append(Pb)
                T_SURFACE.append(Ts)
                TCWV_all.append(TCWV)
                P_all.append(P)
                H_all.append(H)
                TMP_all.append(TMP)
                RH_all.append(RH)
                O3_all.append(O3)
        # spatial interpolation
        if method == 'linear':
            T_SURFACE = [
             spatial_interpolation(self.locs, values, (self.lat, self.lon))
             for values in self._split_array(T_SURFACE)
            ]
            TCWV_all = [
             spatial_interpolation(self.locs, values, (self.lat, self.lon))
             for values in self._split_array(TCWV_all)
            ]
            P_all = self._spatial_interpolate(P_all)
            H_all = self._spatial_interpolate(H_all)
            TMP_all = self._spatial_interpolate(TMP_all)
            RH_all = self._spatial_interpolate(RH_all)
            O3_all = self._spatial_interpolate(O3_all)

        # temporal interpolation
        Ts = temporal_interpolation([0, 6], T_SURFACE, self.delta)
        TCWV = temporal_interpolation([0, 6], TCWV_all, self.delta)
        P = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(P_all)))
        H = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(H_all)))
        TMP = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(TMP_all)))
        RH = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(RH_all)))
        O3 = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(O3_all)))
        layer_number = len(P)

        LOG.debug(
         f'Lon = {lon_0:>10.6f} Lat = {lat_0:>10.6f} Name = PyIRT_Produced Ts = {Ts:>10.3f}\n'
        )
        LOG.debug(
         f'Date_Time = {acq_time.strftime("%Y/%m/%d, %H:%M:%S")} TCWV = {TCWV:>.3f} g/cm^2\n'
        )
        LOG.debug(
         'Height(km)  Pressure(hPa)     AirTemp(K)          RH(%)    Ozone(g/kg)\n'
        )
        Card2C = f'   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(layer_number - 1, -1, -1):
            LOG.debug(
             f'{H[lyr]:>10.4f}{P[lyr]:>15.4f}{TMP[lyr]:>15.4f}{RH[lyr]:>15.4f}{O3[lyr]:>15.8f}\n'
            )
            Card2C += f"{H[lyr]:>10.3f}{P[lyr]:>10.3f}{TMP[lyr]:>10.3f}{RH[lyr]:>10.3e}{0:>10.3e}{O3[lyr]:>10.3e}{atmosphere_model(abs(lat_0), acq_time.month):>10s}\n"
        LOG.debug("\n")
        params = np.c_[list(reversed(H)),
           list(reversed(P)),
           list(reversed(TMP)),
           list(reversed(RH)),
           list(reversed(O3))]
        return Profile(self.profile_name, acq_time, lat_0, lon_0, Ts, layer_number, params,
           Card2C, TCWV)


    def _spatial_interpolate(self, data):
        return [
         list(
          map(
           lambda x: spatial_interpolation(self.locs, x,
          (self.lat, self.lon)),
           np.transpose(values)))
         for values in self._split_array(data)
        ]

    def _split_array(self, data):
        return [data[0:4], data[4:]]

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
             hours=t * self.configuration['resolution_t'] - delta)
            filename = f"gfs_4_{current.year}{current.month:0>2d}{current.day:0>2d}_{current.hour:0>2d}00_000.grb2"
            tiles_file.append(filename)
            if not os.path.exists(f"{self.ncepfilepath}/{filename}"):
                download_gfs_grib(current)
        return tiles_file, delta

class CFSv2Handler(GFSHandler):
    def __init__(self, proxy, ncepfilepath=CFSv2_DIR):
        self.ncepfilepath = ncepfilepath
        self.profile_name = 'CFSv2'
        self.proxy = proxy
        self.configuration = NWPConfiguration['CFSv2']
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def extract_atm_profile(self, acq_time, lat_0, lon_0, method='linear'):

        """ extract atmpospheric profiles (nearest, linear)
            "linear",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "polynomial",
        fill_value "extrapolate"
        """
        self.acq_time = acq_time
        self.lat = lat_0
        self.lon = lon_0
        self.method = method
        ncepfiles, delta = self._check_ncep_folder(acq_time)
        anl_files = [f'{self.ncepfilepath}{file}' for file in ncepfiles]
        self.layer_number = 0  # layer count
        T_SURFACE = []
        TCWV_all = []
        P_all = []
        H_all = []
        TMP_all = []
        RH_all = []
        O3_all = []
        for grb2filename in anl_files:
            # TMP = defaultdict()  # air temperature(K)
            # RH = defaultdict()  # relative humidity(%)
            # O3 = defaultdict()  # Ozone(g/kg)
            # HGT = defaultdict()  # Height(km)
            ds = xr.open_dataset(grb2filename, engine='cfgrib', backend_kwargs={
                "errors": "ignore",
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa"
                }
            })
            # surface_ds = xr.open_dataset(grb2filename, engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'})
            p = ds.coords['isobaricInhPa'].values
            r = self._interp(ds['r']).values
            # sp = self._interp(surface_ds['sp']).values
            t = self._interp(ds['t']).values
            st = t[0]
            z = self._interp(
             ds['gh']
            ).values * 10 / GRAV_ACC  # convert to gpm geopotential height m s-2
            o3 = self._interp(
             ds['o3mr']).values * 1e3  # convert mg/kg to g/kg
            h = list(map(h2z, z))
            TCWV = -999
            # P_SURFACE.append(sp)
            T_SURFACE.append(st)
            TCWV_all.append(TCWV)
            P_all.append(p)
            H_all.append(h)
            TMP_all.append(t)
            RH_all.append(r)
            O3_all.append(o3)
        # for filename in glob.glob(self.ncepfilepath + '*.idx'):
        #     os.remove(filename)
        # temporal interpolation
        Ts = temporal_interpolation([0, 6], T_SURFACE, delta)
        TCWV = temporal_interpolation([0, 6], TCWV_all, delta)
        P = list(
         map(lambda x: temporal_interpolation([0, 6], x, delta),
          np.transpose(P_all)))
        H = list(
         map(lambda x: temporal_interpolation([0, 6], x, delta),
          np.transpose(H_all)))
        TMP = list(
         map(lambda x: temporal_interpolation([0, 6], x, delta),
          np.transpose(TMP_all)))
        RH = list(
         map(lambda x: temporal_interpolation([0, 6], x, delta),
          np.transpose(RH_all)))
        O3 = list(
         map(lambda x: temporal_interpolation([0, 6], x, delta),
          np.transpose(O3_all)))
        layer_number = len(P)

        LOG.debug(
         f'Lon = {lon_0:>10.6f} Lat = {lat_0:>10.6f} Name = PyIRT_Produced Ts = {Ts:>10.3f}\n'
        )
        LOG.debug(
         f'Date_Time = {acq_time.strftime("%Y/%m/%d, %H:%M:%S")} TCWV = {TCWV:>.3f} g/cm^2\n'
        )
        LOG.debug(
         'Height(km)  Pressure(hPa)     AirTemp(K)          RH(%)    Ozone(g/kg)\n'
        )
        Card2C = f'   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(layer_number):
            LOG.debug(
             f'{H[lyr]:>10.4f}{P[lyr]:>15.4f}{TMP[lyr]:>15.4f}{RH[lyr]:>15.4f}{O3[lyr]:>15.8f}\n'
            )
            Card2C += f"{H[lyr]:>10.3f}{P[lyr]:>10.3f}{TMP[lyr]:>10.3f}{RH[lyr]:>10.3e}{0:>10.3e}{O3[lyr]:>10.3e}{atmosphere_model(abs(lat_0), acq_time.month):>10s}\n"
        LOG.debug("\n")
        params = np.c_[H, P, TMP, RH, O3]
        return Profile(self.profile_name, acq_time, lat_0, lon_0, Ts, layer_number, params,
           Card2C, TCWV)

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
             hours=t * self.configuration['resolution_t'] - delta)
            filename = f'cdas1.{current.year}{current.month:0>2d}{current.day:0>2d}' + f't{current.hour:0>2d}z.pgrbanl.grib2'
            original_filename = f'cdas1.t{current.hour:0>2d}z.pgrbanl.grib2'
            if not os.path.exists(self.ncepfilepath + filename):
                pgrb_tar_file = f"{current.year}/cdas1.{current.year}{current.month:0>2d}{current.day:0>2d}.pgrbanl.tar"
                pgb_tar_filename = os.path.basename(pgrb_tar_file)
                if not os.path.exists(f"{self.ncepfilepath}/{pgb_tar_filename}"):
                    download_ucar_file(self.configuration['url'], pgrb_tar_file, f"{self.ncepfilepath}/{pgb_tar_filename}", self.proxy)
                self._unzip(f"{self.ncepfilepath}/{pgb_tar_filename}", original_filename, filename)
            tiles_file.append(filename)
        return tiles_file, delta

    def _unzip(self, filepath, original_filename, filename):
        try:
            t = tarfile.open(filepath, 'r')
            t.extractall(
             self.ncepfilepath,
             members=[m for m in t.getmembers() if m.name==original_filename])
            shutil.move(f"{self.ncepfilepath}/{original_filename}", f"{self.ncepfilepath}/{filename}")
            t.close()
        except IOError as e:
            LOG.info(f'unzip error occur: {e}')
    def _interp(self, ds):
        return ds.interp(latitude=self.lat, longitude=self.lon, method=self.method)

class ERA5Handler:

    def __init__(self, proxy, ncepfilepath=ERA5_DIR):
        self.ncepfilepath = ncepfilepath
        self.profile_name = 'ERA5'
        self.proxy = proxy
        self.configuration = NWPConfiguration['ERA5']
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def extract_atm_profile(self, acq_time, lat, lon, method='linear'):
        self.acq_time = acq_time
        self.lat = lat
        self.lon = lon
        self.method = method
        single_filename, pressure_filename = download_era5_grib(acq_time, lat, lon)
        s_ds = xr.open_dataset(single_filename, engine='cfgrib')
        p_ds = xr.open_dataset(pressure_filename, engine='cfgrib')
        st = self._interp(s_ds['skt']).values
        sp = self._interp(s_ds['sp']).values
        t_ds = self._interp(p_ds['t'])
        t = t_ds.values
        z = self._interp(p_ds['z']).values / GRAV_ACC # convert to geopotential height m s-2
        o3 = self._interp(p_ds['o3']).values * 1e3 # convert kg/kg to g/kg
        r = self._interp(p_ds['r']).values
        p = t_ds.coords['isobaricInhPa'].values
        # clean up
        # for filename in [single_filename, pressure_filename]:
        #     os.remove(filename)
        layer_number = len(p)
        h = list(map(h2z, z))
        card2c = f'   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(layer_number):
            card2c += f"{h[lyr]:>10.3f}{p[lyr]:>10.3f}{t[lyr]:>10.3f}{r[lyr]:>10.3e}{0:>10.3e}{o3[lyr]:>10.3e}{atmosphere_model(abs(lat), acq_time.month):>10s}\n"
        params = np.c_[h, p, t, r, o3]
        return Profile(self.profile_name, acq_time, lat, lon, st, layer_number, params, card2c)

    def _interp(self, ds):
        return ds.interp(latitude=self.lat, longitude=self.lon, time=self.acq_time, method=self.method)

class DOEHandler(GFSHandler):

    def __init__(self, proxy, ncepfilepath=DOE_DIR):
        self.ncepfilepath = ncepfilepath
        self.configuration = NWPConfiguration['DOE']
        self.proxy = proxy
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def extract_atm_profile(self, acq_time, lat_0, lon_0, method='linear'):
        """ extract atmpospheric profiles (nearest, linear)
        """

        ncepfiles, self.delta = self._check_ncep_folder(acq_time)
        self.layer_number = 0  # layer count
        self.acq_time = acq_time
        self.lat = lat_0
        self.lon = (lon_0 + 360) % 360
        # lat_min = math.copysign(1, lat) * int(
        #     abs(lat) / self.configuration['resolution_lat']
        # ) * self.configuration['resolution_lat']
        # lat_max = lat_min + math.copysign(
        #     1, lat) * self.configuration['resolution_lat']
        lat_min = int(
         self.lat / self.configuration['resolution_lat']
        ) * self.configuration['resolution_lat']
        lat_max = lat_min + self.configuration['resolution_lat']


        lat_nearest = lat_min if abs(self.lat - lat_min) < abs(self.lat -
          lat_max) else lat_max

        lon_min = int(self.lon / self.configuration['resolution_lon']
          ) * self.configuration['resolution_lon']
        lon_max = lon_min + self.configuration['resolution_lon']
        lon_nearest = lon_min if abs(self.lon - lon_min) < abs(self.lon -
          lon_max) else lon_max

        self.locs = list(product(
         (lat_min, lat_max),
         (lon_min, lon_max))) if method != 'nearest' else ((lat_nearest,
         lon_nearest), )
        TAPE5_CARD2C = []
        P_SURFACE = []
        T_SURFACE = []
        TCWV_all = []
        P_all = []
        H_all = []
        TMP_all = []
        RH_all = []
        O3_all = []
        reT = re.compile(r'.*?PRES.*sfc')
        for grb2filename in ncepfiles:
            for lat, lon in self.locs:
                TMP = defaultdict()  # air temperature(K)
                RH = defaultdict()  # relative humidity(%)
                O3 = defaultdict()  # Ozone(g/kg)
                HGT = defaultdict()  # Height(km)

                P = []  # Presure(hPa)
                H = []  # Height(km)
                TCWV = 0
                Ts = 0.0  # underlying surface temperature
                Pb_pattern = r'-match :PRES'  # MSLSA MSLET MSLMA PRMSL
                Ts_pattern = r'-match :TMP:surface'
                HGT_pattern = r'-match :HGT\{0,1\}:[[:digit:]]*[\.]\{0,1\}[[:digit:]]*[[:blank:]]*mb'
                PWAT_pattern = r'-match :PWAT'
                O3_pattern = r'-match :O3MR\{0,1\}:[[:digit:]]*[\.]\{0,1\}[[:digit:]]*[[:blank:]]*mb'
                TMP_pattern = r'-match :TMP:[[:digit:]]*[\.]\{0,1\}[[:digit:]]*[[:blank:]]*mb'
                RH_pattern = r'-match :RH\{0,1\}:[[:digit:]]*[\.]\{0,1\}[[:digit:]]*[[:blank:]]*mb'

                cmd = f'"{WGRIB_EXE}" "{self.ncepfilepath+grb2filename}" "{self.ncepfilepath+grb2filename}" | find {{}} | "{WGRIB_EXE}" -i -nh -text "{self.ncepfilepath+grb2filename}" a.txt'
                cmd = f'"{WGRIB_EXE}" "{self.ncepfilepath+grb2filename}" {{}} -s  -lon {lon} {lat}'
                # for line in run_command(cmd.format(Pb_pattern)):
                #     print(line)
                #     Pb = float(line.split("val=")[1].strip()) / 100  # hPa
                # for line in run_command(cmd.format(Ts_pattern)):
                #     Ts = float(line.split("val=")[1].strip())
                # for line in run_command(cmd.format(PWAT_pattern)):
                #     val = float(line.split("val=")[1].strip())
                #     TCWV = val / 10  # convert kg/m^2 to g/cm^2
                # for line in run_command(cmd.format(O3_pattern)):
                #     O3[float(
                #         line.split("mb:anl")[0].split("O3MR:")
                #         [1].strip())] = float(line.split(
                #             "val=")[1].strip()) * 1000  # convert g/g to g/kg
                # for line in run_command(cmd.format(TMP_pattern)):
                #     TMP[float(
                #         line.split("mb:anl")[0].split("TMP:")
                #         [1].strip())] = float(line.split("val=")[1].strip())
                # for line in run_command(cmd.format(RH_pattern)):
                #     val = float(line.split("val=")[1].strip())
                #     RH[float(line.split("mb:anl")[0].split("RH:")
                #              [1].strip())] = val if val else 1e-16
                # for line in run_command(cmd.format(HGT_pattern)):
                #     val = float(line.split("val=")[1].strip())
                #     HGT[float(
                #         line.split("mb:anl")[0].split("HGT:")
                #         [1].strip())] = val

                P = list(TMP.keys())
                H = list(map(h2z, HGT.values()))
                # H = pressure_to_altitude(P, Ts, Pb)
                TMP = list(TMP.values())
                RH = list(fix_null(RH, P, H).values())
                O3 = list(fix_null(O3, P, H).values())
                # P_SURFACE.append(Pb)
                T_SURFACE.append(Ts)
                TCWV_all.append(TCWV)
                P_all.append(P)
                H_all.append(H)
                TMP_all.append(TMP)
                RH_all.append(RH)
                O3_all.append(O3)
        # spatial interpolation
        if method == 'linear':
            T_SURFACE = [
             spatial_interpolation(self.locs, values, (self.lat, self.lon))
             for values in self._split_array(T_SURFACE)
            ]
            TCWV_all = [
             spatial_interpolation(self.locs, values, (self.lat, self.lon))
             for values in self._split_array(TCWV_all)
            ]
            P_all = self._spatial_interpolate(P_all)
            H_all = self._spatial_interpolate(H_all)
            TMP_all = self._spatial_interpolate(TMP_all)
            RH_all = self._spatial_interpolate(RH_all)
            O3_all = self._spatial_interpolate(O3_all)

        # temporal interpolation
        Ts = temporal_interpolation([0, 6], T_SURFACE, self.delta)
        TCWV = temporal_interpolation([0, 6], TCWV_all, self.delta)
        P = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(P_all)))
        H = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(H_all)))
        TMP = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(TMP_all)))
        RH = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(RH_all)))
        O3 = list(
         map(lambda x: temporal_interpolation([0, 6], x, self.delta),
          np.transpose(O3_all)))
        layer_number = len(P)

        LOG.debug(
         f'Lon = {lon_0:>10.6f} Lat = {lat_0:>10.6f} Name = PyIRT_Produced Ts = {Ts:>10.3f}\n'
        )
        LOG.debug(
         f'Date_Time = {acq_time.strftime("%Y/%m/%d, %H:%M:%S")} TCWV = {TCWV:>.3f} g/cm^2\n'
        )
        LOG.debug(
         'Height(km)  Pressure(hPa)     AirTemp(K)          RH(%)    Ozone(g/kg)\n'
        )
        Card2C = f'   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(layer_number - 1, -1, -1):
            LOG.debug(
             f'{H[lyr]:>10.4f}{P[lyr]:>15.4f}{TMP[lyr]:>15.4f}{RH[lyr]:>15.4f}{O3[lyr]:>15.8f}\n'
            )
            Card2C += f"{H[lyr]:>10.3f}{P[lyr]:>10.3f}{TMP[lyr]:>10.3f}{RH[lyr]:>10.3e}{0:>10.3e}{O3[lyr]:>10.3e}{atmosphere_model(abs(lat_0), acq_time.month):>10s}\n"
        LOG.debug("\n")
        params = np.c_[list(reversed(H)),
           list(reversed(P)),
           list(reversed(TMP)),
           list(reversed(RH)),
           list(reversed(O3))]
        return Profile(self.profile_name, acq_time, lat_0, lon_0, Ts, layer_number, params,
           Card2C, TCWV)
    def _loads(self, gribs):
        n_levels = len(gribs)
        levels = np.array([grb_element['level'] for grb_element in gribs])
        indexes = np.argsort(levels)[::-1]  # highest pressure first
        objects = []
        for i in range(n_levels):
            objects.append(gribs[indexes[i]])
        dicts = {
         'object': objects,
         'units': objects[0]['units'],
         'levels': levels[indexes]
        }
        return dicts

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
             hours=t * self.configuration['resolution_t'] - delta)
            filename = 'pgb.anl.'+ current.strftime('%Y%m%d%H') + '.grib'
            if not os.path.exists(self.ncepfilepath + filename):
                pgb_tar_file = f'pgb-anl/{current.year}/pgb.anl.{current.year}{current.month:0>2d}.tar'
                pgb_tar_filename = os.path.basename(pgb_tar_file)
                if not os.path.exists(f"{self.ncepfilepath}/{pgb_tar_filename}"):
                    download_ucar_file(self.configuration['url'], pgb_tar_file, f"{self.ncepfilepath}/{pgb_tar_filename}", self.proxy)
                self._unzip(f"{self.ncepfilepath}/{pgb_tar_filename}", filename)
            tiles_file.append(filename)
        return tiles_file, delta

    def _unzip(self, filepath, filename):
        try:
            t = tarfile.open(filepath, 'r')
            t.extractall(
             self.ncepfilepath,
             members=[m for m in t.getmembers() if m.name==filename])
            t.close()
        except IOError as e:
            LOG.info(f'unzip error occur: {e}')

class GDASHandler(GFSHandler):

    def __init__(self, proxy, ncepfilepath=GDAS_DIR):
        self.ncepfilepath = ncepfilepath
        self.profile_name = 'GDAS'
        self.proxy = proxy
        self.configuration = NWPConfiguration['GDAS']
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
             hours=t * self.configuration['resolution_t'] - delta)
            if datetime(current.year, current.month, current.day, current.hour) > datetime(2007, 12, 6, 6):
                ext = 'grib2'
            else:
                ext = 'grib1'
            file = f"{ext}/{current.year}/{current.year}.{current.month:0>2d}/fnl_{current.year}{current.month:0>2d}{current.day:0>2d}_{current.hour:0>2d}_00.{ext}"
            filename = os.path.basename(file)
            tiles_file.append(filename)
            if not os.path.exists(f"{self.ncepfilepath}/{filename}"):
                download_ucar_file(self.configuration['url'], file,
                 f"{self.ncepfilepath}/{filename}",
                 self.proxy)
        return tiles_file, delta

class GDAS25Handler(GFSHandler):

    def __init__(self, proxy, ncepfilepath=GDAS25_DIR):
        self.ncepfilepath = ncepfilepath
        self.profile_name = 'GDAS25'
        self.proxy = proxy
        self.configuration = NWPConfiguration['GDAS25']
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        tiles_file = []
        for t in range(2):
            current = acq_time + timedelta(
             hours=t * self.configuration['resolution_t'] - delta)
            if datetime(current.year, current.month, current.day, current.hour) < datetime(2015, 7, 8):
                raise Exception('{acq_time} not include in time range of GDAS/FNL 0.25 degree dataset.')
            file = f"{current.year}/{current.year}{current.month:0>2d}/gdas1.fnl0p25.{current.year}{current.month:0>2d}{current.day:0>2d}{current.hour:0>2d}.f00.grib2"
            filename = os.path.basename(file)
            tiles_file.append(filename)
            if not os.path.exists(f"{self.ncepfilepath}/{filename}"):
                download_ucar_file(self.configuration['url'], file,
                 f"{self.ncepfilepath}/{filename}",
                 self.proxy)
        return tiles_file, delta

class JRA55Handler:

    def __init__(self, proxy, ncepfilepath=JRA55_DIR):
        self.ncepfilepath = ncepfilepath
        self.profile_name = 'JRA55'
        self.proxy = proxy
        self.configuration = NWPConfiguration['JRA55']
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")

    def extract_atm_profile(self, acq_time, lat, lon, method='linear'):

        """ extract atmpospheric profiles (nearest, linear)
            "linear",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "polynomial",
        fill_value "extrapolate"
        """
        self.acq_time = acq_time
        self.lat = lat
        self.lon = lon
        self.method = method
        ncepfiles = self._check_ncep_folder(acq_time)
        anl_files = [
         f'{self.ncepfilepath}{file}' for file in ncepfiles
         if 'fcst_p125' not in file
        ]
        fcst_files = [
         f'{self.ncepfilepath}{file}' for file in ncepfiles
         if 'fcst_p125' in file
        ]
        rh_files = [
         f'{self.ncepfilepath}{file}' for file in ncepfiles
         if 'anl_p125.052_rh' in file
        ]
        anl_ds = merge_grib_files(anl_files)
        fcst_ds = merge_grib_files(fcst_files)
        # rh_ds = merge_grib_files(rh_files)
        # r = self._interp(rh_ds['r']).values
        st = self._interp(anl_ds['t2m']).values
        sp = self._interp(anl_ds['sp']).values
        t_ds = self._interp(anl_ds['t'])
        t = t_ds.values
        z = self._interp(
         anl_ds['gh']
        ).values * 10 / GRAV_ACC  # convert to gpm geopotential height m s-2
        o3 = self._interp(
         fcst_ds['ozonehbl']).values * 1e-3  # convert mg/kg to g/kg
        r = self._interp(anl_ds['r']).values
        h = list(map(h2z, z))
        r[np.isnan(r)] = 0
        p = t_ds.coords['isobaricInhPa'].values
        # r = list(fix_null({r_ps[i]: rh for i, rh in enumerate(r)}, p, h).values())
        layer_number = len(p)
        h = list(map(h2z, z))
        card2c = f'   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(layer_number - 1, -1, -1):
            card2c += f"{h[lyr]:>10.3f}{p[lyr]:>10.3f}{t[lyr]:>10.3f}{r[lyr]:>10.3e}{0:>10.3e}{o3[lyr]:>10.3e}{atmosphere_model(abs(lat), acq_time.month):>10s}\n"
        params = np.c_[list(reversed(h)),
           list(reversed(p)),
           list(reversed(t)),
           list(reversed(r)),
           list(reversed(o3))]
        return Profile(self.profile_name,
           acq_time,
           lat,
           lon,
           st,
           layer_number,
           params,
           card2c)

    def _check_ncep_folder(self, acq_time):
        hours = acq_time.hour + acq_time.minute / 60.0 + acq_time.second / 3600.0
        delta = hours % self.configuration['resolution_t']
        tiles_file = []
        download_files = []
        for t in range(2):
            current = acq_time + timedelta(
             hours=t * self.configuration['resolution_t'] - delta)
            start_str = current.strftime("%Y%m0100")
            total_days = calendar.monthrange(current.year, current.month)[1]
            end_str = current.strftime("%Y%m") + str(total_days).zfill(2) + '18'
            pressure_hgt = f"anl_p125/{current.year}/anl_p125.007_hgt.{start_str}_{end_str}"
            pressure_tmp = f"anl_p125/{current.year}/anl_p125.011_tmp.{start_str}_{end_str}"
            pressure_rh = f"anl_p125/{current.year}/anl_p125.052_rh.{start_str}_{end_str}"
            pressure_ozone = f"fcst_p125/{current.year}/fcst_p125.237_ozone.{start_str}_{end_str}"
            single_ts = f"anl_land125/{current.year}/anl_land125.145_tsg.{start_str}_{end_str}"
            single_ts = f"anl_surf125/{current.year}/anl_surf125.011_tmp.{start_str}_{end_str}"
            single_ps = f"anl_surf125/{current.year}/anl_surf125.001_pres.{start_str}_{end_str}"
            single_prmsl = f"anl_surf125/{current.year}/anl_surf125.002_prmsl.{start_str}_{end_str}"
            files = [pressure_hgt, pressure_tmp, pressure_rh, pressure_ozone, single_ts, single_ps, single_prmsl]
            for file in files:
                filename = os.path.basename(file)
                if filename not in tiles_file:
                    tiles_file.append(filename)
                if os.path.exists(os.path.join(self.ncepfilepath, filename)) or file in download_files:
                    continue
                download_files.append(file)
        if len(download_files) > 0:
            download_ucar_multi_files(self.configuration['url'],
              download_files, self.ncepfilepath,
              self.proxy)
        return tiles_file

    def _interp(self, ds):
        return ds.interp(latitude=self.lat, longitude=self.lon, time=self.acq_time, method=self.method)

class MERRA2Handler(ERA5Handler):

    def __init__(self, proxy, ncepfilepath=MERRA2_DIR):
        self.ncepfilepath = ncepfilepath
        self.configuration = NWPConfiguration['MERRA2']
        self.profile_name = 'MERRA2'
        self.proxy = proxy
        self.satellite = False  # satellite-borne platform or not
        if not os.path.isdir(os.path.abspath(ncepfilepath)):
            raise ValueError(
             f"directory {ncepfilepath} is not a valid directory!")


    def extract_atm_profile(self, acq_time, lat, lon, method='linear'):
        # “linear”, “nearest”, “zero”, “slinear”, “quadratic”, “cubic”
        self.acq_time = acq_time
        self.lat = lat
        self.lon = lon
        self.method = method
        nc_filename= download_merra2_nc(acq_time, lat, lon, self.proxy)
        ds = xr.open_dataset(nc_filename, engine='netcdf4')
        res = ds.interp(time=acq_time, lat=lat, lon=lon, method=method)
        p = res.coords['lev'].values
        t = res['T'].values
        sp = res['PS'].values
        slp = res['SLP'].values
        qv = res['QV'].values
        o3 = res['O3'].values * 1e3
        z = res['H'].values
        st = t[-1]
        h = list(map(h2z, z))
        mask = np.logical_or(np.isnan(qv), np.isnan(t))
        mask = np.logical_or(np.isnan(o3), mask)
        # qv = list(
        #     fix_null({p[i]: qv[i]
        #               for i, imask in enumerate(np.isnan(qv)) if not imask}, p,
        #              h).values())
        # t = list(
        #     fix_null(
        #         {
        #             p[i]: t[i]
        #             for i, imask in enumerate(np.isnan(t)) if not imask
        #         }, p, h).values())
        # o3 = list(
        #     fix_null(
        #         {
        #             p[i]: o3[i]
        #             for i, imask in enumerate(np.isnan(o3)) if not imask
        #         }, p, h).values())
        r = np.array(sh2rh(p, qv, t)) * 1e2
        h = np.array(h)[~mask]
        p = p[~mask]
        t = t[~mask]
        r = r[~mask]
        o3 = o3[~mask]
        layer_number = len(p)
        card2c = f'   {layer_number}    0    0                           0.0    0     1.000    28.964  !card2c\n'
        for lyr in range(layer_number):
            card2c += f"{h[lyr]:>10.3f}{p[lyr]:>10.3f}{t[lyr]:>10.3f}{r[lyr]:>10.3e}{0:>10.3e}{o3[lyr]:>10.3e}{atmosphere_model(abs(lat), acq_time.month):>10s}\n"
        params = np.c_[h, p, t, r, o3]
        return Profile(self.profile_name, acq_time, lat, lon, st, layer_number, params, card2c)

    def _interp(self, ds):
        return ds.interp(latitude=self.lat, longitude=self.lon, time=self.acq_time, method=self.method)
