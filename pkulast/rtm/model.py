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
Model wrapper.
=====

Run atmospheric radiative transfer model.

"""

# from cProfile import label
from ast import Mult
import os
import re
import glob
import pytz
import logging
import time
import shutil
import logging
import subprocess
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from subprocess import check_output
from multiprocessing import freeze_support
from datetime import timezone
from pkulast.utils.collections import get_sun_position, get_elevation, is_day
from pkulast.utils.spectrum import get_effective_quantity, modtran_convolve, convert_spectral_domain, modtran_convolve_wn, modtran_resample
from pkulast.surface.spectrum import SpectralAlbedo, Spectra
from pkulast.config import *

LOG = logging.getLogger(__name__)


def _get_work_dir():
    work_dir = os.path.join(RTM_MODTRAN_DIR, "users", os.getenv('JUPYTERHUB_USER'))
    if not os.path.exists(work_dir):
        try:
            original_umask = os.umask(0)
            os.makedirs(work_dir, 0o777)
        finally:
            os.umask(original_umask)
    return work_dir

def _write_mod5_in(mode='correction'):
    """ Modtran input tape5
	"""
    work_dir = _get_work_dir()
    with open(work_dir + "/mod5root.in", "w") as mod5in:
        if mode == 'simulation':
            mod5in.write(f"{SIMUL_DIR}.tp5")
        elif mode == 'transmittance':
            mod5in.write(f"{TRANS_DIR}.tp5")
        else:
            mod5in.write(f"{UWARD_DIR}.tp5\n{DWARD_DIR}.tp5\n{TRANS_DIR}.tp5")


def _single_user_execute(mode_dir, mode_id):
    for chn in glob.glob(TP_DIR + "*.chn"):
        if mode_id in chn:
            os.remove(chn)
    cwd = os.getcwd()
    work_dir = _get_work_dir()
    folders = [
        'DATA',
        'MOD.exe',
        'run_mod.sh',
    ]  #['DATA', 'mie', 'novam']
    for folder in folders:
        if not os.path.exists(os.path.join(work_dir, folder)):
            os.symlink(os.path.join(RTM_MODTRAN_DIR, folder),
                    os.path.join(work_dir, folder))
    os.chdir(work_dir)
    process = subprocess.run(['./run_mod.sh'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            input='',
                            encoding='ascii')
    output =  process.stdout #.decode() #check_output('./run_mod.sh', shell=False).decode()
    if 'Error' in output:
        # os.chdir(cwd)
        raise Exception(
            output.split(
                '********************************************************')
            [-1].strip())
    while not os.path.exists(mode_dir + '.chn'):
        continue
    os.chdir(cwd)

class ModtranWrapper:
    """ modtran wrapper.
	"""

    def __init__(self, rsr, is_satellite, flight_altitude=None):
        if is_satellite:
            flight_altitude = 100  #km
        if flight_altitude is None and not is_satellite:
            raise ValueError(
                'Non-satellite-borne sensor need provide flight altitude!')
        self.filter = rsr
        self.spec_alb = SpectralAlbedo()
        self.is_satellite = is_satellite
        self.flight_altitude = flight_altitude
        self.mode = None
        if self.filter.unit == 'um':
            self.factor = 1e-3
        elif self.filter.unit == 'nm':
            self.factor = 1
        else:
            raise ValueError('unsupported unit used!')

    @property
    def tape_dir(self):
        return TP_DIR

    def plot(self,
             mode=None,
             show_span=True,
             auto_scale=False,
             interval=[0, 25],
             show_response=False,
             alpha=0.3,):
        x = None
        y = None
        title = ''
        if mode is None:
            mode = self.mode
        if mode is None:
            raise ValueError("A simulation should be done first!")
        if mode == 'C':
            ret = np.genfromtxt(UWARD_DIR + '.plt')
            x = ret[:, 0] * self.factor * 1e3
            if not np.array_equal(sorted(x), x):
                x = np.array(list(reversed(x)), dtype=np.float64)
                y = np.array(list(reversed(y)), dtype=np.float64)
            x, indices = np.unique(x, return_index=True)
            y = ret[:, 1][indices] * 1e4
            l, r = interval
            wl = x
            index = np.logical_and(wl > l, wl < r)
            y_min = max(min(y[index]), 0)
            y_max = min(max(y[index]), 1e5)
            title = 'Atmospheric Parameters'
            ret = np.genfromtxt(DWARD_DIR + '.plt')
            dward_y = ret[:, 1][indices] * 1e4
            y_min = min(min(dward_y[index]), y_min)
            y_max = max(max(dward_y[index]), y_max)
            ret = np.genfromtxt(TRANS_DIR + '.plt')
            trans_y = ret[:, 1][indices]
            fig, left_axis = plt.subplots(figsize=[12, 10])
            left_axis.axis([l, r, y_min, y_max])
            left_axis.plot(x[index], y[index], color='r', label="Upwelling Radiance")
            left_axis.plot(x[index], dward_y[index], color='g', label="Downwelling Radiance")
            if auto_scale:
                left_axis.set_ylim(0.95 * y_min, 1.05 * y_max)
            else:
                left_axis.set_ylim(0, 1.05 * y_max)
            left_axis.legend(loc="upper left")
            left_axis.set_ylabel('radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)')
            left_axis.set_xlabel(f'wavelength({self.filter.get_unit_symbol()})')
            right_axis = left_axis.twinx()
            right_axis.plot(x[index], trans_y[index], color='b', label="Transmittance")
            if show_span:
                for i, band in enumerate(self.filter):
                    band.plot_span(right_axis, self.filter.rev_camp(i), alpha=alpha)
                    if show_response:
                        band.plot_response(right_axis, self.filter.cmap(i), alpha=0.5)
            right_axis.legend(loc="upper right")
            if show_response:
                right_axis.set_ylim(0, 1.05)
                right_axis.set_ylabel('spectral response(%)')
            plt.title(f'{title}')
            plt.show()
            return
        if mode == 'S':
            ret = np.genfromtxt(SIMUL_DIR + '.plt')
            title = 'Simulated Radiance'
            ylabel = 'radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)'
            x = ret[:, 0] * self.factor * 1e3
            y = ret[:, 1] * 1e4
        elif mode == 'U':
            ret = np.genfromtxt(UWARD_DIR + '.plt')
            x = ret[:, 0] * self.factor * 1e3
            y = ret[:, 1] * 1e4
            title = 'Upwelling Radiance'
            ylabel = 'radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)'
        elif mode == 'D':
            ret = np.genfromtxt(DWARD_DIR + '.plt')
            title = 'Downwelling Radiance'
            ylabel = 'radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)'
            x = ret[:, 0] * self.factor * 1e3
            y = ret[:, 1] * 1e4
        elif mode == 'T':
            ret = np.genfromtxt(TRANS_DIR + '.plt')
            title = 'Transmittance'
            ylabel = 'transmittance(%)'
            x = ret[:, 0] * self.factor * 1e3
            y = ret[:, 1]
        else:
            raise ValueError(f'mode {mode} is not supported')
        fig, left_axis = plt.subplots(figsize=[12, 10])
        l, r = interval
        if not np.array_equal(sorted(x), x):
            x = np.array(list(reversed(x)), dtype=np.float64)
            y = np.array(list(reversed(y)), dtype=np.float64)
        x, indices = np.unique(x, return_index=True)
        y = y[indices]
        wl = x
        index = np.logical_and(wl > l, wl < r)
        y_min = max(min(y[index]), 0)
        y_max = min(max(y[index]), 1e5)
        left_axis.axis([l, r, y_min, y_max])
        left_axis.plot(x[index], y[index], color='r', label=title)
        if auto_scale:
            left_axis.set_ylim(0.95 * y_min, 1.05 * y_max)
        else:
            left_axis.set_ylim(0, 1.05 * y_max)
        left_axis.legend(loc="upper left")
        left_axis.set_ylabel(ylabel)
        left_axis.set_xlabel(f'wavelength({self.filter.get_unit_symbol()})')
        if show_span:
            right_axis = left_axis.twinx()
            for i, band in enumerate(self.filter):
                band.plot_span(right_axis, self.filter.rev_camp(i), alpha=alpha)
                if show_response:
                    band.plot_response(right_axis, self.filter.cmap(i))
            right_axis.legend(loc="upper right")
            if show_response:
                right_axis.set_ylim(0, 1.05)
                right_axis.set_ylabel('spectral response(%)')
        plt.title(f'{title}')
        plt.show()

    def plot_chn(self, mode=None, show_span=False, auto_scale=False, run_index=0):
        '''
        mode:[S, C, U, D, T]
        '''
        x = None
        y = None
        title = ''
        if self.filter.unit == 'um':
            factor = 1e-3
        elif self.filter.unit == 'nm':
            factor = 1
        else:
            raise ValueError('unsupported unit used!')
        if mode is None:
            mode = self.mode
        if mode is None:
            raise ValueError("A simulation should be done first!")
        x = ['B{}'.format(iband) for iband in self.filter.selected_bands]
        if mode == 'C':
            ret = _read_chn(self.filter.selected_band_count,
                         saved_path=None,
                         factor=factor,
                         mode='correction')
            y = ret['uward'][run_index]
            y_min = max(min(y), 0)
            y_max = min(max(y), 1e5)
            ylabel = 'radiance($Wm^{-2}\mu m^{-1}sr^{-1}$)'
            dward_y = ret['dward'][run_index]
            y_min = min(min(dward_y), y_min)
            y_max = max(max(dward_y), y_max)
            trans_y = ret['trans'][run_index]
            fig, left_axis = plt.subplots(figsize=[12, 10])
            left_axis.plot(x, y, color='r', label='Upwelling Radiance', marker='s')
            left_axis.plot(x, dward_y, color='g', label='Downwelling Radiance', marker='^')
            if auto_scale:
                left_axis.set_ylim(0.95 * y_min, 1.05 * y_max)
            else:
                left_axis.set_ylim(0, 1.05 * y_max)
            left_axis.legend(loc="upper left")
            left_axis.set_ylabel('radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)')
            left_axis.set_xlabel(f'Band Number')
            right_axis = left_axis.twinx()
            left_axis.plot(x, trans_y, color='b', label='Transmittance')
            if show_span:
                for i, band in enumerate(self.filter):
                    band.plot_span(right_axis, self.filter.rev_camp(i), alpha=0.4)
            right_axis.set_ylim(0, 1.05)
            right_axis.legend(loc="upper right")
            right_axis.set_ylabel('transmittance(%)')
            plt.title(f'{title}')
            plt.show()
            return
        if mode == 'S':
            ret = _read_chn(self.filter.selected_band_count,
                         saved_path=None,
                         factor=factor,
                         mode='simulation')
            title = 'Simulated Radiance'
            ylabel = 'radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)'
            y = ret['simul'][run_index]
        elif mode == 'U':
            ret = _read_chn(self.filter.selected_band_count,
                         saved_path=None,
                         factor=factor,
                         mode='correction')
            y = ret['uward'][run_index]
            title = 'Upwelling Radiance'
            ylabel = 'radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)'
        elif mode == 'D':
            ret = _read_chn(self.filter.selected_band_count,
                         saved_path=None,
                         factor=factor,
                         mode='correction')
            y = ret['dward'][run_index]
            title = 'Downwelling Radiance'
            ylabel = 'radiance($\mathrm{Wm^{-2}\mu m^{-1}sr^{-1}}$)'
        elif mode == 'T':
            ret = _read_chn(self.filter.selected_band_count,
                         saved_path=None,
                         factor=factor,
                         mode='transmittance')
            title = 'Transmittance'
            ylabel = 'transmittance(%)'
            y = ret['trans'][run_index]
        else:
            raise ValueError(f'mode {mode} is not supported')
        # plt.figure(figsize=[12, 10])
        fig, left_axis = plt.subplots(figsize=[12, 10])

        left_axis.plot(x, y, color='black', label=title, marker='s')
        if auto_scale:
            left_axis.set_ylim(0.95 * np.min(y), 1.05 * np.max(y))
        else:
            left_axis.set_ylim(0, 1.05 * np.max(y))
        left_axis.legend(loc="upper left")
        left_axis.set_ylabel(ylabel)
        left_axis.set_xlabel(f'Band Number')
        if show_span:
            right_axis = left_axis.twinx()
            for i, band in enumerate(self.filter):
                band.plot_span(right_axis, self.filter.rev_camp(i), alpha=0.4)
            right_axis.set_ylim(0, 1.05)
            right_axis.legend(loc="upper right")
            right_axis.set_ylabel('spectral response(%)')
        plt.title(f'{title}')
        plt.show()

    def load_tape7(self, mode='S'):
        if mode == 'S':  #* 1e4
            filename = SIMUL_DIR + '.tp7'
            title = 'Simulated Radiance'
        elif mode == 'U':
            filename = UWARD_DIR + '.tp7'
            title = 'Upwelling Radiance'
        elif mode == 'D':
            filename = DWARD_DIR + '.tp7'
            title = 'Downwelling Radiance'
        else:
            filename = TRANS_DIR + '.tp7'
            title = 'Transmittance'
        LOG.info(f'loading tape7 of {title}')
        return load_tape7(filename)

    def transmittance(self,
                      profiles,
                      vza=0,
                      vaa=0,
                      mult=False,
                      ground_altitude=None,
                      relative_altitude=None,
                      out_file=None):
        ''' Run modtran
		ITYPE:
			1: horizontal path
			2: Vertical/slant path between 2 Heights
			3: Vertical/slant path to space/ground
		IEMSCT:
			0: transmittance only
		IMULT:
			0: without multiple scattering
			+-1: with multiple scattering
		'''
        self.mode = 'T'
        _write_mod5_in('transmittance')
        srf_path = self.filter.save()
        profile_count = len(profiles)
        max_wv, min_wv = self.filter.get_wv_bounds()
        max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
        simf = open(TRANS_DIR + '.tp5', "w")
        if self.is_satellite:
            ITYPE = 3
        else:
            ITYPE = 2
        IEMSCT = 0
        if mult:
            if self.is_satellite:
                IMULT = -1
            else:
                IMULT = 1
        else:
            IMULT = 0
        # write tape5
        for i, profile in enumerate(profiles):
            # 'TMF 7    2    1   -1    0    0    3    3    3    3    0    1    0   0.001    0.0   !card1\n'
            card_1 = f'TMF 7{ITYPE:5d}{IEMSCT:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
            card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'
            filter_str = f'{srf_path}\n'
            if ground_altitude is None:
                current_ground_altitude = get_elevation(
                    profile.latitude, profile.longitude)
            else:
                current_ground_altitude = ground_altitude
            if current_ground_altitude <= profile.H[0]:
                current_ground_altitude = profile.H[0] + 0.001
                LOG.info(
                    f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
                )
            card_2 = f'    1    0    0    0    0    0    23.000     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
            card_2c = profile.get_card2c()
            if relative_altitude is None:
                current_relative_altitude = self.flight_altitude - current_ground_altitude
            else:
                current_relative_altitude = relative_altitude
            H1ALT = current_relative_altitude + current_ground_altitude
            H2ALT = 0 if ITYPE == 3 else profile.H[0]
            card_3 = f'{H1ALT:>10.3f}{H2ALT:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'
            if IEMSCT == 2:
                card_3a1 = "   12    2{:>5}    0\n".format(
                    profile.acq_time.timetuple().tm_yday)
                acq_time = pytz.utc.localize(profile.acq_time)
                acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
                SZA, SAA = get_sun_position(profile.latitude,
                                            profile.longitude, acq_time)
                LOG.info(f'SZA: {SZA}, SAA: {SAA}')
                los = SAA - vaa# relative solar azimuth [-180, 180]
                LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
                card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                    LOS, SZA)
            else:
                card_3a1 = ''
                card_3a2 = ''
            card_4 = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
            card_5 = '    1 !card5\n'
            if i >= profile_count - 1:
                card_5 = '    0 !card5'

            tp5 = card_1 + card_1a + filter_str + card_2 + \
             card_2c + card_3 + card_3a1 + card_3a2  + card_4+ card_5
            simf.write(tp5)
        simf.close()
        _single_user_execute(TRANS_DIR, TRANS_ID)
        if self.filter.unit == 'um':
            factor = 1e-3
        elif self.filter.unit == 'nm':
            factor = 1
        else:
            raise ValueError('unsupported unit used!')
        return _read_chn(self.filter.selected_band_count,
                         out_file,
                         factor=factor,
                         mode='transmittance')

    def simulation(self,
                   profiles,
                   tbound=None,
                   vza=0,
                   vaa=0,
                   mult=True,
                   albedo=1.0,
                   dt=0,
                   include_solar=False,
                   ground_altitude=None,
                   relative_altitude=None,
                   spectra=None,
                   spectra_id=None,
                   out_file=None,
                   full_spectra=False):
        ''' Run modtran
		ITYPE:
			1: horizontal path
			2: Vertical/slant path between 2 Heights
			3: Vertical/slant path to space/ground
		mode:
			0: transmittance only
			1: thermal radiance (no sun / moon )
			2: thermal plus solar/lunar radiance
			3: directly solar/lunar irradiance
			4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
		mult:
			0: without multiple scattering
			+-1: with multiple scattering
		'''
        self.mode = 'S'
        _write_mod5_in('simulation')
        srf_path = self.filter.save()
        profile_count = len(profiles)
        max_wv, min_wv = self.filter.get_wv_bounds()
        max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
        simf = open(SIMUL_DIR + '.tp5', "w")
        if self.is_satellite:
            ITYPE = 3
        else:
            ITYPE = 2
        if include_solar:
            IEMSCT = 2
        else:
            IEMSCT = 1
        if mult:
            if self.is_satellite:
                IMULT = -1
            else:
                IMULT = 1
        else:
            IMULT = 0
        if spectra_id is None:
            if spectra != None:
                albedo = self.spec_alb.add_spec_alb(spectra.x, spectra.y,
                                                    spectra.name)
        else:
            albedo = spectra_id
        # write tape5
        for i, profile in enumerate(profiles):
            # 'KMF 7    {ITYPE}    {IEMSCT}   {IMULT}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n
            acq_time = pytz.utc.localize(profile.acq_time)
            if IEMSCT == 2 and not is_day(profile.latitude, profile.longitude,
                                          acq_time):
                iemsct = 1
                LOG.info(
                    f"profile {i} is a nightime profile, can't calculate sun radiance, change the mode to TIR"
                )
            else:
                iemsct = IEMSCT
            if tbound is None:
                LOG.info(f"tbound not set and user lower boundary temperature")
                current_tbound = profile.TMP[0] + dt
            else:
                current_tbound = tbound
                current_tbound += dt
            card_1 = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0{current_tbound:>8.3f}{albedo:>7.2f}   !card1\n'  # CARD1
            card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'
            filter_str = f'{srf_path}\n'
            if ground_altitude is None:
                current_ground_altitude = get_elevation(
                    profile.latitude, profile.longitude)
            else:
                current_ground_altitude = ground_altitude
            if current_ground_altitude <= profile.H[0]:
                current_ground_altitude = profile.H[0] + 0.001
                LOG.info(
                    f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
                )
            if profile.VIS < 0:  # visibility
                vis = 23
            else:
                vis = profile.VIS
            card_2 = f'    1    0    0    0    0    0{vis:>10.3f}     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
            card_2c = profile.get_card2c()
            if relative_altitude is None:
                current_relative_altitude = self.flight_altitude - current_ground_altitude
            else:
                current_relative_altitude = relative_altitude
            H1ALT = current_relative_altitude + current_ground_altitude
            H2ALT = 0 if ITYPE == 3 else profile.H[0]
            if ITYPE != 3:
                if abs(vza) < 90:
                    # looking down
                    H2ALT = profile.H[0]
                elif abs(vza) <= 180:
                    # looking up
                    H2ALT = 100
                else:
                    raise ValueError("vza range from 0 to 180 (degree)")

            card_3 = f'{H1ALT:>10.3f}{H2ALT:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'
            if iemsct == 2:
                card_3a1 = "   12    2{:>5}    0\n".format(
                    profile.acq_time.timetuple().tm_yday)
                acq_time = pytz.utc.localize(profile.acq_time)
                acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
                SZA, SAA = get_sun_position(profile.latitude,
                                            profile.longitude, acq_time)
                LOG.info(f'SZA: {SZA}, SAA: {SAA}')
                los = SAA - vaa# relative solar azimuth [-180, 180]
                LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
                card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                    LOS, SZA)
            else:
                card_3a1 = ''
                card_3a2 = ''
            if full_spectra:
                card_4 = f'   400.000 33333.000       1.0       2.0RM              F  1             !card4\n'
            else:
                card_4 = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
            card_5 = '    1 !card5\n'
            if i >= profile_count - 1:
                card_5 = '    0 !card5'

            tp5 = card_1 + card_1a + filter_str + card_2 + \
             card_2c + card_3 + card_3a1 + card_3a2  + card_4+ card_5
            simf.write(tp5)

        simf.close()
        _single_user_execute(SIMUL_DIR, SIMUL_ID)
        self.spec_alb.restore()
        return _read_chn(self.filter.selected_band_count,
                         out_file,
                         factor=self.factor,
                         mode='simulation')

    def correction(self,
                   profiles,
                   mult=True,
                   include_solar=False,
                   vza=0,
                   vaa=0,
                   ground_altitude=None,
                   relative_altitude=None,
                   method='d',
                   out_file=None):
        ''' Run modtran, and return upward, downward, trans
		ITYPE:
			1: horizontal path
			2: Vertical/slant path between 2 Heights
			3: Vertical/slant path to space/ground
		IEMSCT:
			0: transmittance only
			1: thermal radiance (no sun / moon )
			2: thermal plus solar/lunar radiance
			3: directly solar/lunar irradiance
			4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
		IMULT:
			0: without multiple scattering
			+-1: with multiple scattering
		'''
        self.mode = 'C'
        _write_mod5_in()
        srf_path = self.filter.save()
        profile_count = len(profiles)
        upf = open(UWARD_DIR + '.tp5', "w")
        downf = open(DWARD_DIR + '.tp5', "w")
        tranf = open(TRANS_DIR + '.tp5', "w")
        if self.is_satellite:
            ITYPE = 3
        else:
            ITYPE = 2
        if include_solar:
            IEMSCT = 2
        else:
            IEMSCT = 1
        if mult:
            if self.is_satellite:
                IMULT = -1
            else:
                IMULT = 1
        else:
            IMULT = 0
        max_wv, min_wv = self.filter.get_wv_bounds()
        max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
        # write tape5
        for i, profile in enumerate(profiles):
            acq_time = pytz.utc.localize(profile.acq_time)
            if IEMSCT == 2 and not is_day(profile.latitude, profile.longitude,
                                          acq_time):
                iemsct = 1
                LOG.info(
                    f"profile {i} is a nightime profile, can't calculate sun radiance, change the mode to TIR"
                )
            else:
                iemsct = IEMSCT
            # 'KMF 7    {ITYPE}    {IEMSCT}   {IMULT}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n
            card_1 = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001   0.00   !card1\n'  # CARD1
            card_1_down = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
            card_1_downR = f'TMF 7{ITYPE:5d}{iemsct:>5d}    1    0    0    0    0    0    0    0    1    0   0.001    1.0   !card1\n'  # CARD1
            card_1_tran = f'K   7{ITYPE:5d}    0    0    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1

            card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'

            filter_str = f'{srf_path}\n'

            if ground_altitude is None:
                current_ground_altitude = get_elevation(
                    profile.latitude, profile.longitude)
            else:
                current_ground_altitude = ground_altitude
            if current_ground_altitude < profile.H[0]:
                current_ground_altitude = profile.H[0] + 0.001
            card_2 = f'    1    0    0    0    0    0    23.000     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
            card_2c = profile.get_card2c()
            if relative_altitude is None:
                current_relative_altitude = self.flight_altitude - current_ground_altitude
            else:
                current_relative_altitude = relative_altitude
            H1ALT = current_relative_altitude + current_ground_altitude
            # H2ALT_U = 0 if ITYPE == 3 else profile.H[0] modified to ground altitude, 20220630, by Zhu
            H2ALT_U = 0 if ITYPE == 3 else current_ground_altitude
            H2ALT_D = 0 if ITYPE == 3 else 100
            card_3_uptran = f'{H1ALT:>10.3f}{H2ALT_U:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'

            card_3_down = f'{current_ground_altitude:>10.3f}{H2ALT_D:>10.3f}    53.000     0.000     0.000     0.000    0          0.000 !card3\n'

            card_3_downR = f'{current_ground_altitude:>10.3f}{H2ALT_U:>10.3f}   180.000     0.000     0.000     0.000    0          0.000 !card3\n'

            if iemsct == 2:
                card_3a1 = "   12    2{:>5}    0\n".format(
                    profile.acq_time.timetuple().tm_yday)
                # acq_time = pytz.utc.localize(profile.acq_time)
                acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
                SZA, SAA = get_sun_position(profile.latitude,
                                            profile.longitude, acq_time)
                LOG.info(f'SZA: {SZA}, SAA: {SAA}')
                los = SAA - vaa# relative solar azimuth [-180, 180]
                LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
                card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                    LOS, SZA)
            else:
                card_3a1 = ''
                card_3a2 = ''
            card_4_updown = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
            card_4_tran = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0TM              F  1             !card4\n'
            card_5 = '    1 !card5\n'

            if i >= profile_count - 1:
                card_5 = '    0 !card5'

            # UPWARD
            up_tp5 = card_1 + card_1a + filter_str + card_2 + \
             card_2c + card_3_uptran + card_3a1 + card_3a2  + card_4_updown + card_5

            # DOWNWARD
            # 53deg observe downward radiance
            down_tp5 = card_1_down + card_1a + filter_str + card_2 + \
             card_2c + card_3_down + card_3a1 + card_3a2 + card_4_updown + card_5

            # reflect downward radiance
            down_tp5R = card_1_downR + card_1a + filter_str + card_2 + \
             card_2c + card_3_downR + card_3a1 + card_3a2  + card_4_updown + card_5

            # TRANS
            # transmittance
            trans_tp5 = card_1_tran + card_1a + filter_str + card_2 + \
             card_2c + card_3_uptran  +card_4_tran + card_5

            upf.write(up_tp5)
            d_tp5 = down_tp5 if method == 'd' else down_tp5R
            downf.write(d_tp5)
            tranf.write(trans_tp5)
        upf.close()
        downf.close()
        tranf.close()
        # remove all chn files
        for chn in glob.glob(TP_DIR + "/*.chn"):
            if SIMUL_ID not in chn:
                os.remove(chn)
        cwd = os.getcwd()
        work_dir = _get_work_dir()
        folders = [
            'DATA',
            'MOD.exe',
            'run_mod.sh',
        ]  #['DATA', 'mie', 'novam']
        for folder in folders:
            if not os.path.exists(os.path.join(work_dir, folder)):
                os.symlink(os.path.join(RTM_MODTRAN_DIR, folder),
                        os.path.join(work_dir, folder))
        os.chdir(work_dir)
        process = subprocess.run(['./run_mod.sh'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                input='',
                                encoding='ascii')
        output = process.stdout #check_output('./run_mod.sh', shell=False).decode('utf-8')
        if 'Error' in output:
            os.chdir(cwd)
            raise Exception(
                output.split(
                    '********************************************************')
                [-1].strip())
        while not (os.path.exists(UWARD_DIR + '.chn')
                   and os.path.exists(DWARD_DIR + '.chn')
                   and os.path.exists(TRANS_DIR + '.chn')):
            continue
        os.chdir(cwd)
        # output = check_output(f'start /D "{RTM_MODTRAN_DIR}" ./run_mod.sh /WAIT',shell=True)
        if self.filter.unit == 'um':
            factor = 1e-3
        elif self.filter.unit == 'nm':
            factor = 1
        else:
            raise ValueError('unsupported unit used!')
        return _read_chn(self.filter.selected_band_count,
                         out_file,
                         factor=factor)


def run_modtran(profiles,
                rsr,
                is_satellite,
                flight_altitude=None,
                ground_altitude=None,
                relative_altitude=None,
                mult=False,
                include_solar=False,
                vza=0,
                vaa=0,
                method='d',
                out_file=None):
    ''' Run modtran
	ITYPE:
		1: horizontal path
		2: Vertical/slant path between 2 Heights
		3: Vertical/slant path to space/ground
	IEMSCT:
		0: transmittance only
		1: thermal radiance (no sun / moon )
		2: thermal plus solar/lunar radiance
		3: directly solar/lunar irradiance
		4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
	IMULT:
		0: without multiple scattering
		+-1: with multiple scattering
	'''
    _write_mod5_in()
    srf_path = rsr.save()
    profile_count = len(profiles)
    upf = open(UWARD_DIR + '.tp5', "w")
    downf = open(DWARD_DIR + '.tp5', "w")
    tranf = open(TRANS_DIR + '.tp5', "w")
    if is_satellite:
        flight_altitude = 100
        ITYPE = 3
    else:
        ITYPE = 2
    if flight_altitude is None and relative_altitude is None:
        raise ValueError(
            'Non-satellite-borne sensor need provide flight altitude!')

    if include_solar:
        IEMSCT = 2
    else:
        IEMSCT = 1

    if mult:
        if is_satellite:
            IMULT = -1
        else:
            IMULT = 1
    else:
        IMULT = 0
    max_wv, min_wv = rsr.get_wv_bounds()
    max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
    # write tape5
    for i, profile in enumerate(profiles):
        acq_time = pytz.utc.localize(profile.acq_time)
        if IEMSCT == 2 and not is_day(profile.latitude, profile.longitude,
                                      acq_time):
            iemsct = 1
            LOG.info(
                f"profile {i} is a nightime profile, can't calculate sun radiance, change the mode to TIR"
            )
        else:
            iemsct = IEMSCT
        card_1 = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
        card_1_down = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
        card_1_downR = f'TMF 7{ITYPE:5d}{iemsct:>5d}    1    0    0    0    0    0    0    0    1    0   0.001    1.0   !card1\n'  # CARD1
        card_1_tran = f'K   7{ITYPE:5d}    0    0    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1

        card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'

        filter_str = f'{srf_path}\n'
        if ground_altitude is None:
            current_ground_altitude = get_elevation(profile.latitude,
                                                    profile.longitude)
        else:
            current_ground_altitude = ground_altitude
        if current_ground_altitude < profile.H[0]:
            current_ground_altitude = profile.H[0] + 0.001
            LOG.info(
                f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
            )
        card_2 = f'    1    0    0    0    0    0    23.000     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'

        card_2c = profile.get_card2c()
        if relative_altitude is None:
            current_relative_altitude = flight_altitude - current_ground_altitude
        else:
            current_relative_altitude = relative_altitude
        H1ALT = current_relative_altitude + current_ground_altitude
        # H2ALT_U = 0 if ITYPE == 3 else profile.H[0] modified to ground altitude, 20220630, by Zhu
        H2ALT_U = 0 if ITYPE == 3 else current_ground_altitude
        H2ALT_D = 0 if ITYPE == 3 else 100
        # H2ALT_D = 0 if ITYPE == 2 else 100
        card_3_uptran = f'{H1ALT:>10.3f}{H2ALT_U:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'

        card_3_down = f'{current_ground_altitude:>10.3f}{H2ALT_D:>10.3f}    53.000     0.000     0.000     0.000    0          0.000 !card3\n'

        card_3_downR = f'{current_ground_altitude:>10.3f}{H2ALT_U:>10.3f}   180.000     0.000     0.000     0.000    0          0.000 !card3\n'
        if iemsct == 2:
            card_3a1 = "   12    2{:>5}    0\n".format(
                profile.acq_time.timetuple().tm_yday)
            acq_time = pytz.utc.localize(profile.acq_time)
            # acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
            SZA, SAA = get_sun_position(profile.latitude, profile.longitude,
                                        acq_time)
            LOG.info(f'SZA: {SZA}, SAA: {SAA}')
            los = SAA - vaa# relative solar azimuth [-180, 180]
            LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
            card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                LOS, SZA)
        else:
            card_3a1 = ''
            card_3a2 = ''
        card_4_updown = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
        card_4_tran = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0TM              F  1             !card4\n'
        card_5 = '    1 !card5\n'

        if i >= profile_count - 1:
            card_5 = '    0 !card5'

        # UPWARD
        up_tp5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3_uptran + card_3a1 + card_3a2  + card_4_updown + card_5

        # DOWNWARD
        # 53deg observe downward radiance
        down_tp5 = card_1_down + card_1a + filter_str + card_2 + \
         card_2c + card_3_down + card_3a1 + card_3a2 + card_4_updown + card_5

        # reflect downward radiance
        down_tp5R = card_1_downR + card_1a + filter_str + card_2 + \
         card_2c + card_3_downR + card_3a1 + card_3a2  + card_4_updown + card_5

        # TRANS
        # transmittance
        trans_tp5 = card_1_tran + card_1a + filter_str + card_2 + \
         card_2c + card_3_uptran  +card_4_tran + card_5

        upf.write(up_tp5)
        d_tp5 = down_tp5 if method == 'd' else down_tp5R
        downf.write(d_tp5)
        tranf.write(trans_tp5)
    upf.close()
    downf.close()
    tranf.close()
    # remove all chn files
    for chn in glob.glob(TP_DIR + "/*.chn"):
        if SIMUL_ID not in chn:
            os.remove(chn)
    cwd = os.getcwd()
    work_dir = _get_work_dir()
    folders = [
        'DATA',
        'MOD.exe',
        'run_mod.sh',
    ]  #['DATA', 'mie', 'novam']
    for folder in folders:
        if not os.path.exists(os.path.join(work_dir, folder)):
            os.symlink(os.path.join(RTM_MODTRAN_DIR, folder),
                    os.path.join(work_dir, folder))
    os.chdir(work_dir)
    # process = subprocess.run(['./run_mod.sh'],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         input='',
    #                         encoding='ascii')
    process = subprocess.run(['./MOD.exe'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             input='',
                             encoding='ascii')
    output = process.stdout # check_output('./run_mod.sh', shell=False).decode('utf-8')
    if 'Error' in output:
        os.chdir(cwd)
        raise Exception(
            output.split(
                '********************************************************')
            [-1].strip())
    while not (os.path.exists(UWARD_DIR + '.chn')
               and os.path.exists(DWARD_DIR + '.chn')
               and os.path.exists(TRANS_DIR + '.chn')):
        continue
    os.chdir(cwd)
    # output = check_output(f'start /D "{RTM_MODTRAN_DIR}" ./run_mod.sh /WAIT',shell=True)
    if rsr.unit == 'um':
        factor = 1e-3
    elif rsr.unit == 'nm':
        factor = 1
    else:
        raise ValueError('unsupported unit used!')
    return _read_chn(rsr.selected_band_count, out_file, factor=factor)


def _read_chn(band_count, saved_path, factor=1, mode='correction'):
    ''' Read .chn file to extract upwelling-radiance/downwelling-radiance/transmittance.
	'''
    SPLIT_INDEX = band_count + 5
    result = {}
    if mode == 'correction':
        with open(UWARD_DIR + '.chn',
                  "r") as upf, open(DWARD_DIR + '.chn', "r") as downf, open(
                      TRANS_DIR + '.chn', "r") as tranf:
            uprad_lines = list(upf.readlines())
            downrad_lines = list(downf.readlines())
            tran_lines = list(tranf.readlines())

            u_all = []
            d_all = []
            t_all = []

            u = []
            d = []
            t = []
            row_index = 0
            for i in range(len(tran_lines)):
                row_index += 1
                if row_index < 6:
                    continue
                else:
                    u.append(
                        float(uprad_lines[i].strip().split()[3]) * SCALAR *
                        factor)
                    d.append(
                        float(downrad_lines[i].strip().split()[3]) * SCALAR *
                        factor)
                    t.append(1 - float(tran_lines[i].strip().split()[2]))
                if (row_index == SPLIT_INDEX):
                    u_all.append(u)
                    d_all.append(d)
                    t_all.append(t)
                    u = []
                    d = []
                    t = []
                    row_index = 0
        result = {
            'uward': np.array(u_all),
            'dward': np.array(d_all),
            'trans': np.array(t_all)
        }
        if saved_path is not None:
            np.savetxt(saved_path, np.c_[u_all, d_all, t_all])
            # print("Upwelling-Radiance(W/(sr·cm^2·um))/Downwelling-Radiance(W/(sr·cm^2·um))/Transmittance saved at {}".format(saved_path))
    elif mode == 'transmittance':
        while not os.path.exists(TRANS_DIR + '.chn'):
            continue
        with open(TRANS_DIR + '.chn', "r") as tranf:
            tran_lines = list(tranf.readlines())
            t_all = []
            t = []
            row_index = 0
            for i in range(len(tran_lines)):
                row_index += 1
                if row_index < 6:
                    continue
                else:
                    t.append(1 - float(tran_lines[i].strip().split()[2]))
                if row_index == SPLIT_INDEX:
                    t_all.append(t)
                    t = []
                    row_index = 0
        result = {'trans': np.array(t_all)}
        if saved_path is not None:
            np.savetxt(saved_path, np.array(t_all).transpose())
            # print("Transmittance saved at {}".format(saved_path))
    else:
        while not os.path.exists(SIMUL_DIR + '.chn'):
            continue
        with open(SIMUL_DIR + '.chn', "r") as simf:
            sim_lines = list(simf.readlines())
            s_all = []
            s = []
            row_index = 0
            for i in range(len(sim_lines)):
                row_index += 1
                if row_index < 6:
                    continue
                else:
                    s.append(
                        float(sim_lines[i].strip().split()[3]) * SCALAR *
                        factor)
                if row_index == SPLIT_INDEX:
                    s_all.append(s)
                    s = []
                    row_index = 0
        result = {'simul': np.array(s_all)}
        # result = np.array(s_all)
        if saved_path is not None:
            np.savetxt(saved_path, np.array(s_all).transpose())
            # print("Simulation-Radiance(W/(sr·cm^2·um)) saved at {}".format(saved_path))
    return result


def load_tape7(filename):
    ModtranReader.loadtape7(filename)


def load_chn(rsr, chn_path, mode):
    return ModtranReader.loadchn(rsr, chn_path, mode=mode)


def load_chn_atm(rsr, atm_path=RTM_MODTRAN_DIR + 'tape/'):
    result = {
        'uward': ModtranReader.loadchn(rsr, atm_path + 'uward.chn', mode='r'),
        'dward': ModtranReader.loadchn(rsr, atm_path + 'dward.chn', mode='r'),
        'trans': ModtranReader.loadchn(rsr, atm_path + 'trans.chn', mode='t'),
    }
    return result


def fixHeaders(instr):
    """
    Modifies the column header string to be compatible with numpy column lookup.
    """
    intab = "+--[]@"
    outtab = "pmmbba"
    translate_table = str.maketrans(intab, outtab)
    rtnstring = instr.translate(translate_table)
    return rtnstring


def fixHeadersList(headcol):
    headcol = [fixHeaders(strn) for strn in headcol]
    return headcol


class ModtranReader:

    colspec = None
    IEMSCT = None
    data = None

    def __init__(self, filename='') -> None:
        self.filename = filename

    @property
    def mode(self):
        if self.IEMSCT == 0:  # Transmmittance
            return 'Transmmittance'
        elif self.IEMSCT == 1:  # Thermal radiance
            return 'Thermal radiance'
        elif self.IEMSCT == 2:  # Radiance with/or multiple scattering
            return 'Radiance with/or multiple scattering'
        elif self.IEMSCT == 3:  # Solar irradiance
            return 'Solar irradiance'
        else:
            raise ValueError("Wrong inputs for IEMSCT!")

    @classmethod
    def cols(cls, IEMSCT=None):
        if cls.IEMSCT is not None and IEMSCT is None:
            IEMSCT = cls.IEMSCT
        if IEMSCT == 0:  # Transmmittance
            return [
                'FREQ_CM-1', 'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS',
                'O3_TRANS', 'TRACE_TRANS', 'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT',
                'AER+CLD_TRANS', 'HNO3_TRANS', 'AER+CLD_abTRNS', '-LOG_COMBIN',
                'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS', 'N2O_TRANS', 'O2_TRANS',
                'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS', 'SO2_TRANS',
                'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS',
                'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS',
                'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS',
                'CCL4_TRANS', 'N2O5_TRANS'
            ]  #MODTRAN5.2.2 don't valid ['H2-H2_TRANS','H2-HE_TRANS','H2-CH4_TRANS', 'CH4-CH4_TRANS']
        elif IEMSCT == 1:  # Thermal radiance
            return [
                'FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS',
                'SOL_SCAT', 'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD',
                'REF_SOL', 'SOL@OBS', 'DEPTH', 'DIR_EM', 'TOA_SUN',
                'BBODY_T[K]'
            ]  # don't valid ['SOL_SCAT', 'SING_SCAT', 'DRCT_RFLT', 'REF_SOL', 'SOL@OBS', 'TOA_SUN']
        elif IEMSCT == 2:  # Radiance with/or multiple scattering
            return [
                'FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS',
                'SOL_SCAT', 'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD',
                'REF_SOL', 'SOL@OBS', 'DEPTH', 'DIR_EM', 'TOA_SUN',
                'BBODY_T[K]'
            ]
        elif IEMSCT == 3:  # Solar irradiance
            return ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']
        else:
            raise ValueError("Wrong inputs for IEMSCT!")

    @classmethod
    def integral(cls,
                 rsr,
                 plt_path,
                 profile_count,
                 saved_path=None,
                 trapz=False,
                 mode='r',
                 line=32934):
        ''' utilizing the plt data to integral effective quantity.
        '''
        import dask.array as da
        from dask import delayed
        import dask.dataframe as dd
        if mode == 'r':
            factor = 1e4
        else:
            factor = 1
        data = dd.read_csv(plt_path,
                           header=None).to_dask_array(lengths=True).reshape(
                               profile_count, line)
        if trapz:
            func = extract_data_trapz
        else:
            func = extract_data
        arr = [
            da.from_delayed(value=delayed(func)(data[i], rsr,
                                                rsr.selected_bands),
                            shape=(rsr.selected_band_count, ),
                            dtype=np.float64) for i in range(profile_count)
        ]
        quan = da.stack(arr)
        quan *= factor
        quan = quan.compute()
        if saved_path is not None:
            np.savetxt(saved_path, quan)
            # print("Integral quantity saved at {}".format(saved_path))
        return quan

    @classmethod
    def loadplt(cls, plt_path, profile_count, saved_path=None, mode='r',line=32934):
        import dask.array as da
        from dask import delayed
        import dask.dataframe as dd
        if mode == 'r':
            factor = 1e4
        else:
            factor = 1
        data = dd.read_csv(plt_path,
                           header=None).to_dask_array(lengths=True).reshape(
                               profile_count, line)
        func = lambda rad: np.array(
            [list(map(float,
                      x.strip().split('    '))) for x in rad])
        arr = [
            da.from_delayed(value=delayed(func)(data[i]),
                            shape=(line, ),
                            dtype=np.float64) for i in range(profile_count)
        ]
        quan = da.stack(arr)
        quan *= factor
        quan = quan.compute()
        if saved_path is not None:
            np.savetxt(saved_path, quan)
            # print("PLT data saved at {}".format(saved_path))
        return quan

    @classmethod
    def loadchn(cls, rsr, chn_path, saved_path=None, mode='r'):
        '''load chn file.
        mode: radiance(r)/transmittance(t)
        '''
        band_number = rsr.selected_band_count
        if rsr.unit == 'um':
            factor = 1e-3
        elif rsr.unit == 'nm':
            factor = 1
        else:
            raise ValueError('unsupported unit used!')
        SPLIT_INDEX = band_number + 5
        result = {}
        if mode == 't':
            if not os.path.exists(chn_path):
                raise OSError("tape5 file not exists!")
            with open(chn_path, "r") as tranf:
                tran_lines = list(tranf.readlines())
                t_all = []
                t = []
                row_index = 0
                for i in range(len(tran_lines)):
                    row_index += 1
                    if row_index < 6:
                        continue
                    else:
                        t.append(1 - float(tran_lines[i].strip().split()[2]))
                    if row_index == SPLIT_INDEX:
                        t_all.append(t)
                        t = []
                        row_index = 0
            result = np.array(t_all)
            if saved_path is not None:
                np.savetxt(saved_path, np.array(t_all).transpose())
                # print("Transmittance saved at {}".format(saved_path))
        elif mode == 'r':
            while not os.path.exists(chn_path):
                continue
            with open(chn_path, "r") as simf:
                sim_lines = list(simf.readlines())
                s_all = []
                s = []
                row_index = 0
                for i in range(len(sim_lines)):
                    row_index += 1
                    if row_index < 6:
                        continue
                    else:
                        s.append(
                            float(sim_lines[i].strip().split()[3]) * SCALAR *
                            factor)
                    if row_index == SPLIT_INDEX:
                        s_all.append(s)
                        s = []
                        row_index = 0
            result = np.array(s_all)
            if saved_path is not None:
                np.savetxt(saved_path, np.array(s_all).transpose())
                # print("Radiance(W/(sr·cm^2·um)) saved at {}".format(saved_path))
        else:
            raise ValueError(
                'unsupported mode used! mode include radiance(r)/transmittance(t)'
            )
        return result

    @classmethod
    def loadtape7(cls, filename, colspec=[]):
        infile = open(filename, 'r')
        lines = infile.readlines()  #.strip()
        infile.close()
        if len(lines) < 10:
            LOG.error(f'Error reading file {filename}: too few lines!')
            return None
        cols = []
        res = []
        for line in lines:
            cols.append(line)
            if line.strip() == '-9999.':
                res.append(SpectralQuantity(cls._single_tape7(cols, colspec)))
                cols = []
        return res

    @classmethod
    def _single_tape7(cls, lines, colspec=[]):
        """
        This function reads in the tape7 file from MODerate spectral resolution
        atmospheric TRANsmission (MODTRAN) code, that is used to model the
        propagation of the electromagnetic radiation through the atmosphere. tape7
        is a primary file that contains all the spectral results of the MODTRAN
        run. The header information in the tape7 file contains portions of the
        tape5 information that will be deleted. The header section in tape7 is
        followed by a list of spectral points with corresponding transmissions.
        Each column has a different component of the transmission or radiance. 
        For more detail, see the modtran documentation.
        """
        idata = {}
        colHead = []
        #determine values for MODEL, ITYPE, IEMSCT, IMULT from card 1
        #tape5 input format (presumably also tape7, line 1 format?)
        #format Lowtran7  (13I5, F8.3, F7.0) = (MODEL, ITYPE, IEMSCT, IMULT)
        #format Modtran 4 (2A1, I3, 12I5, F8.3, F7.0) = (MODTRN, SPEED, MODEL, ITYPE, IEMSCT, IMULT)
        #format Modtran 5 (3A1, I2, 12I5, F8.0, A7) = (MODTRN, SPEED, BINARY, MODEL, ITYPE, IEMSCT, IMULT)
        #MODEL = int(lines[0][4])
        #ITYPE = int(lines[0][9])
        IEMSCT = int(lines[0][14])
        if not colspec:
            colspec = cls.cols(IEMSCT)
            cls.colspec = colspec
        else:
            cls.colspec = colspec

        #IMULT = int(lines[0][19])
        # print('filename={0}, IEMSCT={1}'.format(filename,IEMSCT))

        #skip the first few rows that contains tape5 information and leave the
        #header for the different components of transimissions.
        #find the end of the header.
        headline = 0
        while lines[headline].find('FREQ') < 0:
            headline = headline + 1

        #some files has only a single text column head, while others have two
        # find out what the case is for this file and concatenate if necessary
        line1 = lines[headline]  # alway header 1
        line2 = lines[headline + 1]  # maybe data, maybe header 2
        line3 = lines[headline + 2]  # definately data

        #see if there is a second header line
        p = re.compile('[a-df-zA-DF-Z]+')
        line2found = True if p.search(line2) is not None else False

        #modtran 4 does not use underscores
        line1 = line1.replace('TOT TRANS', 'TOT_TRANS')
        line1 = line1.replace('PTH THRML', 'PTH_THRML')
        line1 = line1.replace('THRML SCT', 'THRML_SCT')
        line1 = line1.replace('SURF EMIS', 'SURF_EMIS')
        line1 = line1.replace('SOL SCAT', 'SOL_SCAT')
        line1 = line1.replace('SING SCAT', 'SING_SCAT')
        line1 = line1.replace('GRND RFLT', 'GRND_RFLT')
        line1 = line1.replace('DRCT RFLT', 'DRCT_RFLT')
        line1 = line1.replace('TOTAL RAD', 'TOTAL_RAD')
        line1 = line1.replace('REF SOL', 'REF_SOL')

        colcount = 0
        colEnd = []
        #strip newline from the data line
        linet = line3.rstrip()

        idx = 0
        while idx < len(linet):
            while linet[idx].isspace():
                idx += 1
                if idx == len(linet):
                    break
            while not linet[idx].isspace():
                idx += 1
                if idx == len(linet):
                    break
            colEnd.append(idx + 1)

        colSrt = [0] + [v - 1 for v in colEnd[:-1]]
        collim = list(zip(colSrt, colEnd))

        # iemsct=3 has a completely messed up header, replace with this
        if IEMSCT == 3:
            colHead1st = ' '.join(
                ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH'])
        else:
            # each entry in collim defines the slicing start and end for each col, including leading whitepace
            # missing columns may have headers that came through as single long string,
            # now remove by splitting on space and taking the last one
            colHead1st = [
                line1[lim[0]:lim[1] - 1].split()[-1].strip() for lim in collim
            ]
        colHead2nd = [
            line2[lim[0]:lim[1] - 1].split()[-1].strip() for lim in collim
        ]

        # if colHead2nd[0].find('CM') >= 0:
        if line2found:
            colHead = [
                h1 + '_' + h2 for (h1, h2) in zip(colHead1st, colHead2nd)
            ]
            deltaHead = 1
        else:
            colHead = colHead1st
            deltaHead = 0

        #different IEMSCT values have different column formats
        # some cols have headers and some are empty.
        # for IEMSCT of 0 and 2 the column headers are correct and should work as is.
        #for IEMSCT of 1 the following columns are empty and must be deleted from the header
        if IEMSCT == 1:
            removeIEMSCT1 = [
                'SOL_SCAT', 'SING_SCAT', 'DRCT_RFLT', 'REF_SOL', 'SOL@OBS',
                'TOA_SUN'
            ]
            colHead = [x for x in colHead if x not in removeIEMSCT1]

        if IEMSCT == 3:
            colHead = ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']

        # build a new data set with appropriate column header and numeric data
        #change all - and +  to alpha to enable table lookup
        colHead = fixHeadersList(colHead)

        s = ' '.join(colHead) + '\n'
        # now append the numeric data, ignore the original header and last row in the file
        s = s + ''.join(lines[headline + 1 + deltaHead:-1])

        #read the string in from a StringIO in-memory file
        # print(s.encode('utf-8'))
        from io import BytesIO
        lines = np.genfromtxt(BytesIO(s.encode('utf-8')),
                              dtype=None,
                              names=True)

        #extract the wavenumber col as the first column in the new table
        coldata = lines[fixHeaders(colspec[0])].reshape(-1, 1)
        # then append the other required columns
        for colname in colspec[1:]:
            coldata = np.hstack(
                (coldata, lines[fixHeaders(colname)].reshape(-1, 1)))

        res = defaultdict()
        for index, col in enumerate(colspec):
            res[col] = coldata[:, index]
        return res


class SpectralQuantity:
    default_resolution = 1
    default_window = 15

    def __init__(self,
                 data_dict,
                 resolution=1,
                 window=Modtran_Sampling_Resolution):
        self.dict_ = defaultdict()
        self.default_window = self.window = window
        self.default_resolution = self.resolution = resolution
        self.spectral_dict = data_dict
        self._convert_spectral_density()

    @property
    def cols(self):
        return list(self.dict_.keys())

    def resample(self, resolution=1, window=Modtran_Sampling_Resolution):
        self.window = window
        self.resolution = resolution
        self._convert_spectral_density()

    def interp(self, rsr, col):
        fwhm = self.get_resolution(rsr)
        self.resample(fwhm)
        res = rsr.interp(self.wl, self.dict_[col], unit="um")
        self.resample(self.default_resolution, self.default_window)
        return res

    def interpv2(self, rsr, col):
        fwhm = self.get_resolution(rsr)
        self.resample(fwhm)
        res = rsr.effective_value(self.wl, self.dict_[col], unit="um")
        self.resample(self.default_resolution, self.default_window)
        return res

    def get_spectral_quantity(self, col):
        return self.spectral_dict[col]

    def get_resolution(self, rsr):
        intervals = []
        for band in rsr:
            wn = convert_spectral_domain(band.get_wv(unit="um"), "ln")
            intervals.append(np.abs(np.mean(np.diff(wn))))
        fwhm = 2 * np.mean(intervals)
        return fwhm

    def plot(self, cols, min_wl=0.3, max_wl=25, **kwargs):
        if isinstance(cols, str):
            cols = [
                cols,
            ]
        indices = np.where((self.wl >= min_wl) & (self.wl <= max_wl))
        plt.figure(figsize=(10, 8))
        for col in cols:
            plt.plot(self.wl[indices],
                     self.dict_[col][indices],
                     label=col,
                     **kwargs)
        plt.legend()
        plt.show()
        plt.close()

    def plot_spectral_quantity(self, cols, min_wl=0.3, max_wl=25, **kwargs):
        if isinstance(cols, str):
            cols = [
                cols,
            ]
        indices = np.where((self.wl >= min_wl) & (self.wl <= max_wl))
        plt.figure(figsize=(10, 8))
        for col in cols:
            plt.plot(self.wl[indices],
                     self.spectral_dict[col][indices],
                     label=col,
                     **kwargs)
        plt.legend()
        plt.show()
        plt.close()

    def _convert_spectral_density(self):
        cols = list(self.spectral_dict.keys())
        wn = self.spectral_dict[cols[0]]
        self.wl = convert_spectral_domain(wn, "nl")
        for col in cols[1:]:
            if "TRANS" in col or col in [
                    "N2_CONT",
                    "H2O_CONT",
                    "MOLEC_SCAT",
                    "AER+CLD_abTRNS",
                    "-LOG_COMBIN",
            ]:
                self.dict_[col], _ = modtran_resample(self.spectral_dict[col],
                                                      1, self.resolution,
                                                      self.window)
            else:
                wl, self.dict_[col] = modtran_convolve_wn(
                    wn, self.spectral_dict[col] * 1e4, self.resolution,
                    self.window)

    def __getitem__(self, col):
        return self.dict_[col]


def extract_data(rad, rsr, selected_band_count):
    ''' extract profile data from da.Array
    '''
    rad = np.array([list(map(float, x.strip().split('    '))) for x in rad])
    spectral_wv, spectral_res = rad[:, 0], rad[:, 1]
    spectral_wv, spectral_res = modtran_convolve(spectral_wv, spectral_res)
    return np.array([
        rsr[iband].interp(spectral_wv, spectral_res, unit='um')
        for iband in list(range(selected_band_count))
    ])


def extract_data_trapz(rad, rsr, selected_band_count):
    ''' extract profile data using trapz from da.Array
    '''
    rad = np.array([list(map(float, x.strip().split('    '))) for x in rad])
    spectral_wv, spectral_res = rad[:, 0], rad[:, 1]
    spectral_wv, spectral_res = modtran_convolve(spectral_wv, spectral_res)
    return np.array([
        interp_trapz(rsr[iband], spectral_wv, spectral_res, unit='um')
        for iband in list(range(selected_band_count))
    ])


def interp_trapz(band, spectral_wv, spectral_res, unit='nm'):
    from scipy.interpolate import interp1d
    if unit == 'um':
        if band.unit == 'nm':
            spectral_wv = spectral_wv * 1e3
    elif unit == 'nm':
        if band.unit == 'um':
            spectral_wv = spectral_wv * 1e-3
    else:
        raise ValueError(f'unsupported unit: {unit}!')
    return get_effective_quantity(band.wv,
                                  interp1d(spectral_wv, spectral_res)(band.wv),
                                  band.response)


class ModtranTIGR946:

    def __init__(self,
                 profile_count=946,
                 MODTRAN946_DIR=RTM_MODTRAN_DIR + 'TIGR946',
                 with_error=False):
        self.dir = MODTRAN946_DIR
        if with_error:
            self.dir = RTM_MODTRAN_DIR + 'TIGR946(wv10%t1)'
        self.profile_count = profile_count

    def integral(self, rsr, saved_path=None, line=32934):
        import dask.array as da
        from dask import delayed
        import dask.dataframe as dd
        selected_band_count = rsr.selected_band_count
        try:
            self.uward = dd.read_csv(
                self.dir + '/uward.plt',
                header=None).to_dask_array(lengths=True).reshape(
                    self.profile_count, line)
            self.dward = dd.read_csv(
                self.dir + '/dward.plt',
                header=None).to_dask_array(lengths=True).reshape(
                    self.profile_count, line)
            self.trans = dd.read_csv(
                self.dir + '/trans.plt',
                header=None).to_dask_array(lengths=True).reshape(
                    self.profile_count, line)
        except:
            raise ValueError("wrong profile count as input, please recheck!")
        arr = da.stack([
            da.from_delayed(value=delayed(self._interp)(data, rsr,
                                                        selected_band_count),
                            shape=(
                                self.profile_count,
                                rsr.selected_band_count,
                            ),
                            dtype=np.float64)
            for data in [self.uward, self.dward, self.trans]
        ])
        arr_np = arr.compute()
        arr_np[:
               2, :, :] *= 1e4  #convert to radiance(W m^{-2} {um}^{-1} sr^{-1})
        if saved_path is not None:
            np.savetxt(saved_path, np.c_[arr_np[0], arr_np[1], arr_np[2]])
            # print("Integral quantity saved at {}".format(saved_path))
        result = {
            'uward': arr_np[0],
            'dward': arr_np[1],
            'trans': arr_np[2],
        }
        return result

    def integral_t(self, rsr, saved_path=None, line=32934):
        import dask.array as da
        from dask import delayed
        import dask.dataframe as dd
        selected_band_count = rsr.selected_band_count
        try:
            self.trans = dd.read_csv(
                self.dir + '/trans.plt',
                header=None).to_dask_array(lengths=True).reshape(
                    self.profile_count, line)
        except:
            raise ValueError("wrong profile count as input, please recheck!")
        arr = da.stack([
            da.from_delayed(value=delayed(self._interp)(data, rsr,
                                                        selected_band_count),
                            shape=(
                                self.profile_count,
                                rsr.selected_band_count,
                            ),
                            dtype=np.float64) for data in [
                                self.trans,
                            ]
        ])
        arr_np = arr.compute()
        if saved_path is not None:
            np.savetxt(saved_path, arr_np[0])
            # print("Integral quantity saved at {}".format(saved_path))
        result = {
            'trans': arr_np[0],
        }
        return result

    def _interp(self, spec, rsr, selected_band_count):
        results = []
        for i in range(self.profile_count):
            data = extract_data(spec[i], rsr, selected_band_count)
            results.append(data)
        return np.array(results)


class ParallelModtranWrapper(ModtranWrapper):
    """ modtran wrapper.
	"""

    def __init__(self, rsr, is_satellite, flight_altitude=None):
        # elevate(show_console=False)
        if is_satellite:
            flight_altitude = 100  #km
        if flight_altitude is None and not is_satellite:
            raise ValueError(
                'Non-satellite-borne sensor need provide flight altitude!')
        self.filter = rsr
        self.is_satellite = is_satellite
        self.flight_altitude = flight_altitude
        if self.filter.unit == 'um':
            self.factor = 1e-3 * SCALAR
        elif self.filter.unit == 'nm':
            self.factor = 1 * SCALAR
        else:
            raise ValueError('unsupported unit used!')

    def plot_chn(self, mode='S', show_span=True, auto_scale=False):
        '''
        mode:[S, U, D, T]
        '''
        x = None
        y = None
        title = ''
        x = self.filter.centers * self.factor * 1e3
        if mode == 'S':
            title = 'Simulated Radiance'
            ylabel = 'radiance($Wm^{-2}\mu m^{-1}sr^{-1}$)'
            y = self.simul
        elif mode == 'U':
            y = self.uward
            title = 'Upwelling Radiance'
            ylabel = 'radiance($Wm^{-2}\mu m^{-1}sr^{-1}$)'
        elif mode == 'D':
            title = 'Downwelling Radiance'
            ylabel = 'radiance($Wm^{-2}\mu m^{-1}sr^{-1}$)'
            y = self.dward
        else:
            title = 'Transmittance'
            ylabel = 'transmittance(%)'
            y = self.trans
        fig, left_axis = plt.subplots(figsize=[12, 10])

        left_axis.plot(x, y, color='r', label=title)
        if auto_scale:
            left_axis.set_ylim(0.95 * np.min(y), 1.05 * np.max(y))
        else:
            left_axis.set_ylim(0, 1.05 * np.max(y))
        left_axis.legend(loc="upper left")
        left_axis.set_ylabel(ylabel)
        left_axis.set_xlabel(f'wavelength({self.filter.get_unit_symbol()})')
        if show_span:
            right_axis = left_axis.twinx()
            for i, band in enumerate(self.filter):
                band.plot_span(right_axis, self.filter.rev_camp(i), alpha=0.4)
            right_axis.set_ylim(0, 1.05)
            right_axis.legend(loc="upper right")
            right_axis.set_ylabel('spectral response(%)')
        plt.title(f'{title}')
        plt.show()

    def transmittance(self,
                      profiles,
                      vza=0,
                      vaa=0,
                      mult=False,
                      ground_altitude=None,
                      relative_altitude=None,
                      out_file=None):
        ''' Run modtran
		'''
        _, tape5_list = make_tape5_trans(profiles,
                                         self.filter,
                                         self.is_satellite,
                                         out_file=None,
                                         vza=vza,
                                         vaa=vaa,
                                         mult=mult,
                                         flight_altitude=self.flight_altitude,
                                         ground_altitude=ground_altitude,
                                         relative_altitude=relative_altitude)
        self.trans = parallel_executor(tape5_list, self.filter, mode='t')
        result = {'trans': self.trans}
        if out_file is not None:
            np.savetxt(out_file, self.trans)
            # print("Transmittance saved at {}".format(out_file))
        return result

    def simulation(self,
                   profiles,
                   tbound=300,
                   vza=0,
                   vaa=0,
                   mult=True,
                   albedo=1.0,
                   dt=0,
                   include_solar=False,
                   ground_altitude=None,
                   relative_altitude=None,
                   spectra=None,
                   out_file=None):
        _, tape5_list = make_tape5(profiles,
                                   self.filter,
                                   self.is_satellite,
                                   out_file=None,
                                   flight_altitude=self.flight_altitude,
                                   ground_altitude=ground_altitude,
                                   relative_altitude=relative_altitude,
                                   dt=dt,
                                   tbound=tbound,
                                   albedo=albedo,
                                   mult=mult,
                                   include_solar=include_solar,
                                   vza=vza,
                                   vaa=vaa,
                                   spectra=spectra)
        self.simul = parallel_executor(tape5_list, self.filter, mode='r')
        result = {'simul': self.simul}
        if out_file is not None:
            np.savetxt(out_file, self.simul)
            # print("Simulation-Radiance(W/(sr·cm^2·um)) saved at {}".format(out_file))
        return result

    def simulation_with_error(self,
                              profiles,
                              tbound=300,
                              vza=0,
                              vaa=0,
                              mult=True,
                              albedo=1.0,
                              dt=0,
                              include_solar=False,
                              ground_altitude=None,
                              relative_altitude=None,
                              spectra=None,
                              out_file=None,
                              vza_error=0.0):
        _, tape5_list = make_tape5_with_error(
            profiles,
            self.filter,
            self.is_satellite,
            out_file=None,
            flight_altitude=self.flight_altitude,
            ground_altitude=ground_altitude,
            relative_altitude=relative_altitude,
            dt=dt,
            tbound=tbound,
            albedo=albedo,
            mult=mult,
            include_solar=include_solar,
            vza=vza,
            vaa=vaa,
            spectra=spectra,
            vza_error=vza_error)
        self.simul = parallel_executor(tape5_list, self.filter, mode='r')
        result = {'simul': self.simul}
        if out_file is not None:
            np.savetxt(out_file, self.simul)
            # print("Simulation-Radiance(W/(sr·cm^2·um)) saved at {}".format(out_file))
        return result

    def correction(self,
                   profiles,
                   mult=True,
                   include_solar=False,
                   vza=0,
                   vaa=0,
                   ground_altitude=None,
                   relative_altitude=None,
                   method='d',
                   out_file=None):
        _, tape5_list = make_tape5_atm(profiles,
                                       self.filter,
                                       self.is_satellite,
                                       out_file=None,
                                       flight_altitude=self.flight_altitude,
                                       ground_altitude=ground_altitude,
                                       relative_altitude=relative_altitude,
                                       mult=mult,
                                       include_solar=include_solar,
                                       vza=vza,
                                       vaa=vaa,
                                       method=method)
        self.uward = parallel_executor(tape5_list[0], self.filter, mode='r')
        self.dward = parallel_executor(tape5_list[1], self.filter, mode='r')
        self.trans = parallel_executor(tape5_list[2], self.filter, mode='t')
        result = {
            'uward': self.uward,
            'dward': self.dward,
            'trans': self.trans
        }
        if out_file is not None:
            np.savetxt(out_file, np.c_[self.uward, self.dward, self.trans])
            # print("Upwelling-Radiance(W/(sr·cm^2·um))/Downwelling-Radiance(W/(sr·cm^2·um))/Transmittance saved at {}".format(out_file))
        return result


def parallel_run_modtran(profiles,
                         rsr,
                         is_satellite,
                         flight_altitude=None,
                         ground_altitude=None,
                         relative_altitude=None,
                         mult=False,
                         include_solar=False,
                         vza=0,
                         vaa=0,
                         method='d',
                         out_file=None):
    _, tape5_list = make_tape5_atm(profiles,
                                   rsr,
                                   is_satellite,
                                   out_file=None,
                                   flight_altitude=flight_altitude,
                                   ground_altitude=ground_altitude,
                                   relative_altitude=relative_altitude,
                                   mult=mult,
                                   include_solar=include_solar,
                                   vza=vza,
                                   vaa=vaa,
                                   method=method)
    uward = parallel_executor(tape5_list[0], rsr, mode='r', out_file=None)
    dward = parallel_executor(tape5_list[1], rsr, mode='r', out_file=None)
    trans = parallel_executor(tape5_list[2], rsr, mode='t', out_file=None)
    result = {'uward': uward, 'dward': dward, 'trans': trans}
    if out_file is not None:
        np.savetxt(out_file, np.c_[uward, dward, trans])
        print(
            "Upwelling-Radiance(W/(sr·cm^2·um))/Downwelling-Radiance(W/(sr·cm^2·um))/Transmittance saved at {}"
            .format(out_file))
    return result


def _compute(chunked_tape5_list, band_count, factor, mode, remove):
    tp5_str = ''
    tp5_dir = ''
    if mode == 'r':
        index = 3
    elif mode == 't':
        index = 2
    else:
        raise ValueError(f"undefined mode: {mode}")
    order = chunked_tape5_list[0]
    for i, tape5 in enumerate(chunked_tape5_list[1]):
        if tape5:
            if i == len(chunked_tape5_list[1]) - 1:
                card_5 = '    0 !card5\n'
            else:
                card_5 = '    1 !card5\n'
            tp5_dir += '_' + str(tape5[0])
            tp5_str += tape5[1] + card_5
    cwd = os.getcwd()
    work_dir = os.path.join(Modtran_Parallel_Dir, os.getenv('JUPYTERHUB_USER'), tp5_dir)
    if not os.path.exists(work_dir):
        try:
            original_umask = os.umask(0)
            os.makedirs(work_dir, 0o777)
        finally:
            os.umask(original_umask)
    with open(os.path.join(work_dir, 'tape5'), 'w') as f:
        f.write(tp5_str)
    for chn in glob.glob(work_dir + "/*.out"):
        os.remove(chn)
    folders = [
        'DATA',
    ]  #['DATA', 'mie', 'novam']
    for folder in folders:
        if not os.path.exists(os.path.join(work_dir, folder)):
            os.symlink(os.path.join(RTM_MODTRAN_DIR, folder),
                       os.path.join(work_dir, folder))
    os.chdir(work_dir)
    process = subprocess.run([RTM_MODTRAN_DIR + '/MOD.exe'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             input='',
                             encoding='ascii')
    while not os.path.exists(os.path.join(work_dir, 'channels.out')):
        continue
    SPLIT_INDEX = band_count + 5
    with open(os.path.join(work_dir, 'channels.out'), "r") as fn:
        lines = list(fn.readlines())
        quan = []
        single_p = []
        row_index = 0
        for i in range(len(lines)):
            row_index += 1
            if row_index < 6:
                continue
            else:
                single_p.append(
                    float(lines[i].strip().split()[index]) * factor)
            if row_index == SPLIT_INDEX:
                quan.append(np.array(single_p, dtype=np.float64))
                single_p = []
                row_index = 0
    os.chdir(cwd)
    for folder in folders:
        os.unlink(os.path.join(work_dir, folder))
    if remove:
        shutil.rmtree(work_dir)
    result = np.stack(quan) if mode == 'r' else 1 - np.stack(quan)
    return order, result


def _rechunk(tape5_list, chunk_size):
    from itertools import zip_longest
    numbered_tape5_list = list(
        zip_longest(*[iter(list(zip(range(len(tape5_list)), tape5_list)))] *
                    chunk_size,
                    fillvalue=None))
    return list(zip(range(len(numbered_tape5_list)), numbered_tape5_list))


def parallel_executor(tape5_list,
                      rsr,
                      mode='r',
                      chunk_size=CHUNK_SIZE,
                      max_workers=MAX_WORKERS,
                      remove=True,
                      out_file=None):
    if mode == 'r':
        if rsr.unit == 'um':
            factor = 1e-3 * SCALAR
        elif rsr.unit == 'nm':
            factor = 1 * SCALAR
        else:
            raise ValueError('unsupported unit used!')
    else:
        factor = 1
    result = _parallel_executor(
        tape5_list,
        rsr.selected_band_count,
        factor,
        mode,
        chunk_size=chunk_size,
        max_workers=max_workers,
        remove=remove,
    )
    if out_file is not None:
        np.savetxt(out_file, result)
        # print(
        #     "modtran result saved at {}".format(out_file))
    return result


def executor_callback(worker):
    worker_exception = worker.exception()
    if worker_exception:
        LOG.exception("worker return exception: {}".format(worker_exception))


def _parallel_executor(tape5_list,
                       band_count,
                       factor,
                       mode,
                       chunk_size=CHUNK_SIZE,
                       max_workers=MAX_WORKERS,
                       remove=True):
    start_time = time.time()
    quantities = []
    orders = []
    if int(len(tape5_list) / chunk_size) < max_workers:
        num = round(len(tape5_list) / max_workers)
        chunk_size = num if num != 0 else 1
    chunked_tape5_list = _rechunk(tape5_list, chunk_size)
    freeze_support()
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers) as executor:
        futures = [
            executor.submit(_compute, item, band_count, factor, mode, remove)
            for item in chunked_tape5_list
        ]
        for future in futures:
            future.add_done_callback(executor_callback)
        for future in concurrent.futures.as_completed(futures):
            order, quan = future.result()
            orders.append(order)
            quantities.append(quan)
    results = [
        quantities[orders.index(i)] for i in range(len(chunked_tape5_list))
    ]
    print("Parallel MODTRAN execution in " + str(time.time() - start_time) +
          " seconds")
    return np.concatenate(results)


def make_tape5(
    profiles,
    rsr,
    is_satellite,
    out_file=None,
    flight_altitude=None,
    ground_altitude=None,
    relative_altitude=None,
    tbound=None,
    dt=0,
    albedo=1.0,
    mult=False,
    include_solar=False,
    vza=0,
    vaa=0,
    spectra=None,
    spectra_id=None,
    vis=None,
    sza=None,
    saa=None,
    daytime=None,
):
    ''' Run modtran
		ITYPE:
			1: horizontal path
			2: Vertical/slant path between 2 Heights
			3: Vertical/slant path to space/ground
		mode:
			0: transmittance only
			1: thermal radiance (no sun / moon )
			2: thermal plus solar/lunar radiance
			3: directly solar/lunar irradiance
			4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
		mult:
			0: without multiple scattering
			+-1: with multiple scattering
        daytime:
            None, True, False
		'''
    single_tp5 = []
    multi_tp5 = []
    spec_alb = SpectralAlbedo()
    srf_path = rsr.save()
    profile_count = len(profiles)
    if is_satellite:
        ITYPE = 3
    else:
        ITYPE = 2
    if include_solar:
        IEMSCT = 2
    else:
        IEMSCT = 1
    if mult:
        if is_satellite:
            IMULT = -1
        else:
            IMULT = 1
    else:
        IMULT = 0

    if spectra_id is None:
        if spectra != None:
            albedo = spec_alb.add_spec_alb(spectra.x, spectra.y, spectra.name)
    else:
        albedo = spectra_id

    max_wv, min_wv = rsr.get_wv_bounds()
    max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
    # write tape5
    for i, profile in enumerate(profiles):
        # 'KMF 7    {ITYPE}    {IEMSCT}   {IMULT}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n
        acq_time = pytz.utc.localize(profile.acq_time)
        if daytime is None:
            if IEMSCT == 2 and not is_day(profile.latitude, profile.longitude,
                                          acq_time):
                iemsct = 1
                LOG.info(
                    f"profile {i} is a nightime profile, can't calculate sun radiance, change the mode to TIR"
                )
            else:
                iemsct = IEMSCT
        else:
            if IEMSCT == 2 and not daytime:
                iemsct = 1
                LOG.info(
                    f"setting all profiles as nightime profile, can't calculate sun radiance, change the mode to TIR"
                )
            else:
                iemsct = IEMSCT
        if tbound is None:
            LOG.info(f"tbound not set and use lower boundary temperature")
            current_tbound = profile.TMP[0] + dt
        else:
            current_tbound = tbound
            current_tbound += dt
        card_1 = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0{current_tbound:>8.3f}{albedo:>7.2f}   !card1\n'  # CARD1
        card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'
        filter_str = f'{srf_path}\n'
        if ground_altitude is None:
            current_ground_altitude = get_elevation(profile.latitude,
                                                    profile.longitude)
        else:
            current_ground_altitude = ground_altitude
        if current_ground_altitude <= profile.H[0]:
            current_ground_altitude = profile.H[0] + 0.001
            LOG.info(
                f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
            )
        if vis is None:
            if profile.VIS < 0:  # visibility
                current_vis = 23
            else:
                current_vis = profile.VIS
        else:
            current_vis = vis
        card_2 = f'    1    0    0    0    0    0{current_vis:>10.3f}     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
        card_2c = profile.get_card2c()
        if relative_altitude is None:
            current_relative_altitude = flight_altitude - current_ground_altitude
        else:
            current_relative_altitude = relative_altitude
        H1ALT = current_relative_altitude + current_ground_altitude
        H2ALT = 0 if ITYPE == 3 else profile.H[0]
        if ITYPE != 3:
            if abs(vza) < 90:
                # looking down
                H2ALT = profile.H[0]
            elif abs(vza) <= 180:
                # looking up
                H2ALT = 100
            else:
                raise ValueError("vza range from 0 to 180 (degree)")

        card_3 = f'{H1ALT:>10.3f}{H2ALT:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'
        if iemsct == 2:
            card_3a1 = "   12    2{:>5}    0\n".format(
                profile.acq_time.timetuple().tm_yday)
            acq_time = pytz.utc.localize(profile.acq_time)
            acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
            SZA, SAA = get_sun_position(profile.latitude, profile.longitude,
                                        acq_time)
            LOG.info(f'SZA: {SZA}, SAA: {SAA}')
            if saa is not None:
                SAA = saa
            if sza is not None:
                SZA = sza
            los = SAA - vaa# relative solar azimuth [-180, 180]
            LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
            card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                LOS, SZA)
        else:
            card_3a1 = ''
            card_3a2 = ''

        card_4 = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
        card_5 = '    1 !card5\n'
        if i >= profile_count - 1:
            card_5 = '    0 !card5'

        tp5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3 + card_3a1 + card_3a2  + card_4+ card_5

        tp5_no_card5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3 + card_3a1 + card_3a2  + card_4
        single_tp5.append(tp5)
        multi_tp5.append(tp5_no_card5)
    if out_file is not None:
        with open(out_file, "w") as tp5f:
            tp5f.write(''.join(single_tp5))
    return out_file, multi_tp5
    # spec_alb.restore()


def make_tape5_with_error(
    profiles,
    rsr,
    is_satellite,
    out_file=None,
    flight_altitude=None,
    ground_altitude=None,
    relative_altitude=None,
    tbound=None,
    dt=0,
    albedo=1.0,
    mult=False,
    include_solar=False,
    vza=0,
    vaa=0,
    spectra=None,
    spectra_id=None,
    vza_error=0,
):
    ''' Run modtran
		ITYPE:
			1: horizontal path
			2: Vertical/slant path between 2 Heights
			3: Vertical/slant path to space/ground
		mode:
			0: transmittance only
			1: thermal radiance (no sun / moon )
			2: thermal plus solar/lunar radiance
			3: directly solar/lunar irradiance
			4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
		mult:
			0: without multiple scattering
			+-1: with multiple scattering
		'''
    single_tp5 = []
    multi_tp5 = []
    spec_alb = SpectralAlbedo()
    srf_path = rsr.save()
    profile_count = len(profiles)
    if is_satellite:
        ITYPE = 3
    else:
        ITYPE = 2
    if include_solar:
        IEMSCT = 2
    else:
        IEMSCT = 1
    if mult:
        if is_satellite:
            IMULT = -1
        else:
            IMULT = 1
    else:
        IMULT = 0

    if spectra_id is None:
        if spectra != None:
            albedo = spec_alb.add_spec_alb(spectra.x, spectra.y, spectra.name)
    else:
        albedo = spectra_id
    max_wv, min_wv = rsr.get_wv_bounds()
    max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
    # write tape5
    vza_errors = np.random.normal(loc=0, scale=vza_error, size=len(profiles))
    for i, profile in enumerate(profiles):
        current_vza = abs(vza + vza_errors[i])
        # 'KMF 7    {ITYPE}    {IEMSCT}   {IMULT}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n
        acq_time = pytz.utc.localize(profile.acq_time)
        if IEMSCT == 2 and not is_day(profile.latitude, profile.longitude,
                                      acq_time):
            iemsct = 1
            LOG.info(
                f"profile {i} is a nightime profile, can't calculate sun radiance, change the mode to TIR"
            )
        else:
            iemsct = IEMSCT
        if tbound is None:
            LOG.info(f"tbound not set and user lower boundary temperature")
            current_tbound = profile.TMP[0] + dt
        else:
            current_tbound = tbound
            current_tbound += dt
        card_1 = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0{current_tbound:>8.3f}{albedo:>7.2f}   !card1\n'  # CARD1
        card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'
        filter_str = f'{srf_path}\n'
        if ground_altitude is None:
            current_ground_altitude = get_elevation(profile.latitude,
                                                    profile.longitude)
        else:
            current_ground_altitude = ground_altitude
        if current_ground_altitude <= profile.H[0]:
            current_ground_altitude = profile.H[0] + 0.001
            LOG.info(
                f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
            )
        if profile.VIS < 0:  # visibility
            vis = 23
        else:
            vis = profile.VIS
        card_2 = f'    1    0    0    0    0    0{vis:>10.3f}     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
        card_2c = profile.get_card2c()
        if relative_altitude is None:
            current_relative_altitude = flight_altitude - current_ground_altitude
        else:
            current_relative_altitude = relative_altitude
        H1ALT = current_relative_altitude + current_ground_altitude
        H2ALT = 0 if ITYPE == 3 else profile.H[0]
        if ITYPE != 3:
            if abs(current_vza) < 90:
                # looking down
                H2ALT = profile.H[0]
            elif abs(current_vza) <= 180:
                # looking up
                H2ALT = 100
            else:
                raise ValueError("vza range from 0 to 180 (degree)")

        card_3 = f'{H1ALT:>10.3f}{H2ALT:>10.3f}{180-current_vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'
        if iemsct == 2:
            card_3a1 = "   12    2{:>5}    0\n".format(
                profile.acq_time.timetuple().tm_yday)
            acq_time = pytz.utc.localize(profile.acq_time)
            acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
            SZA, SAA = get_sun_position(profile.latitude, profile.longitude,
                                        acq_time)
            LOG.info(f'SZA: {SZA}, SAA: {SAA}')
            los = SAA - vaa# relative solar azimuth [-180, 180]
            LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
            card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                LOS, SZA)
        else:
            card_3a1 = ''
            card_3a2 = ''
        card_4 = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
        card_5 = '    1 !card5\n'
        if i >= profile_count - 1:
            card_5 = '    0 !card5'

        tp5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3 + card_3a1 + card_3a2  + card_4+ card_5

        tp5_no_card5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3 + card_3a1 + card_3a2  + card_4
        single_tp5.append(tp5)
        multi_tp5.append(tp5_no_card5)
    if out_file is not None:
        with open(out_file, "w") as tp5f:
            tp5f.write(''.join(single_tp5))
    return out_file, multi_tp5
    # spec_alb.restore()


def make_tape5_trans(profiles,
                     rsr,
                     is_satellite,
                     out_file=None,
                     vza=0,
                     vaa=0,
                     mult=False,
                     flight_altitude=None,
                     ground_altitude=None,
                     relative_altitude=None):
    ''' Run modtran
	ITYPE:
		1: horizontal path
		2: Vertical/slant path between 2 Heights
		3: Vertical/slant path to space/ground
	IEMSCT:
		0: transmittance only
		1: thermal radiance (no sun / moon )
		2: thermal plus solar/lunar radiance
		3: directly solar/lunar irradiance
		4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
	IMULT:
		0: without multiple scattering
		+-1: with multiple scattering
	'''
    srf_path = rsr.save()
    profile_count = len(profiles)
    trans_single_tp5 = []
    trans_multi_tp5 = []
    if is_satellite:
        flight_altitude = 100
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
    max_wv, min_wv = rsr.get_wv_bounds()
    max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
    if flight_altitude is None and relative_altitude is None:
        raise ValueError(
            'Non-satellite-borne sensor need provide flight altitude!')
    # write tape5
    for i, profile in enumerate(profiles):
        # 'TMF 7    2    1   -1    0    0    3    3    3    3    0    1    0   0.001    0.0   !card1\n'
        card_1 = f'TMF 7{ITYPE:5d}{IEMSCT:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
        card_1a = 'FFF  8 0.0   400.000  0.000000          01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'
        filter_str = f'{srf_path}\n'
        if ground_altitude is None:
            current_ground_altitude = get_elevation(profile.latitude,
                                                    profile.longitude)
        else:
            current_ground_altitude = ground_altitude
        if current_ground_altitude <= profile.H[0]:
            current_ground_altitude = profile.H[0] + 0.001
            LOG.info(
                f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
            )
        card_2 = f'    1    0    0    0    0    0    23.000     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'
        card_2c = profile.get_card2c()
        if relative_altitude is None:
            current_relative_altitude = flight_altitude - current_ground_altitude
        else:
            current_relative_altitude = relative_altitude
        H1ALT = current_relative_altitude + current_ground_altitude
        H2ALT = 0 if ITYPE == 3 else profile.H[0]
        card_3 = f'{H1ALT:>10.3f}{H2ALT:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'
        if IEMSCT == 2:
            card_3a1 = "   12    2{:>5}    0\n".format(
                profile.acq_time.timetuple().tm_yday)
            acq_time = pytz.utc.localize(profile.acq_time)
            acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
            SZA, SAA = get_sun_position(profile.latitude, profile.longitude,
                                        acq_time)
            LOG.info(f'SZA: {SZA}, SAA: {SAA}')
            los = SAA - vaa# relative solar azimuth [-180, 180]
            LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
            card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                LOS, SZA)
        else:
            card_3a1 = ''
            card_3a2 = ''
        card_4 = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
        card_5 = '    1 !card5\n'
        if i >= profile_count - 1:
            card_5 = '    0 !card5'

        tp5 = card_1 + card_1a + filter_str + card_2 + \
            card_2c + card_3 + card_3a1 + card_3a2  + card_4+ card_5

        tp5_no_card5 = card_1 + card_1a + filter_str + card_2 + \
            card_2c + card_3 + card_3a1 + card_3a2  + card_4
        # transmittance
        trans_single_tp5.append(tp5)
        trans_multi_tp5.append(tp5_no_card5)
    if out_file:
        with open(f'{out_file}', "w") as tranf:
            tranf.write(''.join(''.join(trans_single_tp5)))
    return out_file, trans_multi_tp5


def make_tape5_atm(profiles,
                   rsr,
                   is_satellite,
                   out_file=None,
                   flight_altitude=None,
                   ground_altitude=None,
                   relative_altitude=None,
                   mult=False,
                   include_solar=False,
                   vza=0,
                   vaa=0,
                   method='d'):
    ''' Run modtran
	ITYPE:
		1: horizontal path
		2: Vertical/slant path between 2 Heights
		3: Vertical/slant path to space/ground
	IEMSCT:
		0: transmittance only
		1: thermal radiance (no sun / moon )
		2: thermal plus solar/lunar radiance
		3: directly solar/lunar irradiance
		4: solar/lunar radiance with no thernam scatter[including thermal path and surface emission]
	IMULT:
		0: without multiple scattering
		+-1: with multiple scattering
	'''
    srf_path = rsr.save()
    profile_count = len(profiles)
    uward_single_tp5 = []
    uward_multi_tp5 = []
    dward_single_tp5 = []
    dward_multi_tp5 = []
    trans_single_tp5 = []
    trans_multi_tp5 = []
    if is_satellite:
        flight_altitude = 100
        ITYPE = 3
    else:
        ITYPE = 2
    if flight_altitude is None and relative_altitude is None:
        raise ValueError(
            'Non-satellite-borne sensor need provide flight altitude!')

    if include_solar:
        IEMSCT = 2
    else:
        IEMSCT = 1

    if mult:
        if is_satellite:
            IMULT = -1
        else:
            IMULT = 1
    else:
        IMULT = 0
    max_wv, min_wv = rsr.get_wv_bounds()
    max_wn, min_wn = int(1e4 / min_wv) + 10, int(1e4 / max_wv) - 10
    # write tape5
    for i, profile in enumerate(profiles):
        acq_time = pytz.utc.localize(profile.acq_time)
        if IEMSCT == 2 and not is_day(profile.latitude, profile.longitude,
                                      acq_time):
            iemsct = 1
            LOG.info(
                f"profile {i} is a nightime profile, can't calculate sun radiance, change the mode to TIR"
            )
        else:
            iemsct = IEMSCT
        card_1 = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
        card_1_down = f'TMF 7{ITYPE:5d}{iemsct:>5d}{IMULT:>5d}    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1
        card_1_downR = f'TMF 7{ITYPE:5d}{iemsct:>5d}    1    0    0    0    0    0    0    0    1    0   0.001    1.0   !card1\n'  # CARD1
        card_1_tran = f'K   7{ITYPE:5d}    0    0    0    0    0    0    0    0    0    1    0   0.001    0.0   !card1\n'  # CARD1

        card_1a = 'FFF  8 0.0   400.000                    01 F T T       0.000      0.00     0.000     0.000     0.000         0   !card1a\n'

        filter_str = f'{srf_path}\n'
        if ground_altitude is None:
            current_ground_altitude = get_elevation(profile.latitude,
                                                    profile.longitude)
        else:
            current_ground_altitude = ground_altitude
        if current_ground_altitude < profile.H[0]:
            current_ground_altitude = profile.H[0] + 0.001
            LOG.info(
                f'ground_altitude below profile bottom height, reset ground_altitude to {current_ground_altitude}km.'
            )
        card_2 = f'    1    0    0    0    0    0    23.000     0.000     0.000     0.000{current_ground_altitude:>10.3f}   !card2\n'

        card_2c = profile.get_card2c()
        if relative_altitude is None:
            current_relative_altitude = flight_altitude - current_ground_altitude
        else:
            current_relative_altitude = relative_altitude
        H1ALT = current_relative_altitude + current_ground_altitude
        # H2ALT_U = 0 if ITYPE == 3 else profile.H[0] modified to ground altitude, 20220630, by Zhu
        H2ALT_U = 0 if ITYPE == 3 else current_ground_altitude
        H2ALT_D = 0 if ITYPE == 3 else 100
        # H2ALT_D = 0 if ITYPE == 2 else 100
        card_3_uptran = f'{H1ALT:>10.3f}{H2ALT_U:>10.3f}{180-vza:>10.3f}     0.000     0.000     0.000    0          0.000 !card3\n'

        card_3_down = f'{current_ground_altitude:>10.3f}{H2ALT_D:>10.3f}    53.000     0.000     0.000     0.000    0          0.000 !card3\n'

        card_3_downR = f'{current_ground_altitude:>10.3f}{H2ALT_U:>10.3f}   180.000     0.000     0.000     0.000    0          0.000 !card3\n'
        if iemsct == 2:
            card_3a1 = "   12    2{:>5}    0\n".format(
                profile.acq_time.timetuple().tm_yday)
            acq_time = pytz.utc.localize(profile.acq_time)
            # acq_time = profile.acq_time.replace(tzinfo=timezone.utc)
            SZA, SAA = get_sun_position(profile.latitude, profile.longitude,
                                        acq_time)
            LOG.info(f'SZA: {SZA}, SAA: {SAA}')
            los = SAA - vaa# relative solar azimuth [-180, 180]
            LOS = los - np.sign(los) * 360 if abs(los) > 180 else los
            card_3a2 = "{:>10.3f}{:>10.3f}     0.000     0.000     0.000     0.000     0.000     0.000\n".format(
                LOS, SZA)
        else:
            card_3a1 = ''
            card_3a2 = ''
        card_4_updown = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0RM              F  1             !card4\n'
        card_4_tran = f'{min_wn:>10.3f}{max_wn:>10.3f}       1.0       2.0TM              F  1             !card4\n'
        card_5 = '    1 !card5\n'

        if i >= profile_count - 1:
            card_5 = '    0 !card5'

        # UPWARD
        up_tp5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3_uptran + card_3a1 + card_3a2  + card_4_updown + card_5
        up_tp5_no_card5 = card_1 + card_1a + filter_str + card_2 + \
         card_2c + card_3_uptran + card_3a1 + card_3a2  + card_4_updown

        # DOWNWARD
        # 53deg observe downward radiance
        down_tp5 = card_1_down + card_1a + filter_str + card_2 + \
         card_2c + card_3_down + card_3a1 + card_3a2 + card_4_updown + card_5

        down_tp5_no_card5 = card_1_down + card_1a + filter_str + card_2 + \
         card_2c + card_3_down + card_3a1 + card_3a2 + card_4_updown

        # reflect downward radiance
        down_tp5R = card_1_downR + card_1a + filter_str + card_2 + \
         card_2c + card_3_downR + card_3a1 + card_3a2  + card_4_updown + card_5

        down_tp5R_no_card5 = card_1_downR + card_1a + filter_str + card_2 + \
         card_2c + card_3_downR + card_3a1 + card_3a2  + card_4_updown

        # TRANS
        # transmittance
        trans_tp5 = card_1_tran + card_1a + filter_str + card_2 + \
         card_2c + card_3_uptran  +card_4_tran + card_5
        trans_tp5_no_card5 = card_1_tran + card_1a + filter_str + card_2 + \
         card_2c + card_3_uptran  +card_4_tran

        d_tp5 = down_tp5 if method == 'd' else down_tp5R
        d_tp5_no_card5 = down_tp5_no_card5 if method == 'd' else down_tp5R_no_card5

        uward_single_tp5.append(up_tp5)
        uward_multi_tp5.append(up_tp5_no_card5)
        dward_single_tp5.append(d_tp5)
        dward_multi_tp5.append(d_tp5_no_card5)
        trans_single_tp5.append(trans_tp5)
        trans_multi_tp5.append(trans_tp5_no_card5)
        # tranf.write(trans_tp5)
    if out_file:
        with open(f'{out_file}.uward.tp5', "w") as upf:
            upf.write(''.join(uward_single_tp5))
        with open(f'{out_file}.dward.tp5', "w") as downf:
            downf.write(''.join(dward_single_tp5))
        with open(f'{out_file}.trans.tp5', "w") as tranf:
            tranf.write(''.join(''.join(trans_single_tp5)))
        output = (f'{out_file}.uward.tp5', f'{out_file}.dward.tp5',
                  f'{out_file}.tran.tp5')
    else:
        output = None
    return output, (uward_multi_tp5, dward_multi_tp5, trans_multi_tp5)



class RTMSimul:

    def __init__(
        self,
        rsr,
        dTs=range(-15, 20, 5),
        emis_lib=DEFAULT_EMIS_LIB,
        profile_lib=None,
        mu=[1, ]
    ) -> None:
        self.rsr = rsr
        self.dTs = list(dTs)
        self.emis_lib = emis_lib
        self.emissivities = None
        self.profile_lib = profile_lib
        self.mu = mu
        self.degrees = np.rad2deg(np.arccos(self.mu))


    def true_simulate(self,
                 selected_type=('tropical', 'mid-latitude', 'polar'),
                 selected_tcwv=[0, 6.3]):
        from pkulast.surface.emissivity import get_band_emissivity, get_spec_emissivity
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, _ = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)

        self.emissivities = get_band_emissivity(self.rsr, self.emis_lib)
        self.spec_emissivities = get_spec_emissivity(self.rsr, self.emis_lib)
        tape5_list = []
        lst_list = []
        tcwv_list = []
        emis_list = []
        mu_list = []
        for vza in self.degrees:
            for wv, emis in self.spec_emissivities:
                for dt in self.dTs:
                    spectra = Spectra(wv, 1 - emis, 100, 'simulated')
                    _, tp5 = make_tape5(profiles,
                                        self.rsr,
                                        True,
                                        out_file=None,
                                        flight_altitude=100,
                                        ground_altitude=0.2,
                                        relative_altitude=None,
                                        tbound=None,
                                        dt=dt,
                                        albedo=1.0,
                                        mult=True,
                                        include_solar=False,
                                        vza=vza,
                                        spectra=spectra)
                    tape5_list.extend(tp5)
                    lst_list.extend([p.TMP[0] + dt for p in profiles])
                    tcwv_list.extend([p.TCWV for p in profiles])
                    emis_list.extend([emis for _ in profiles])
                    mu_list.extend([vza for _ in profiles])
                # break
        radiance = parallel_executor(tape5_list, self.rsr)
        bt = self.rsr.r2bt(radiance)
        lst = np.array(lst_list, dtype=np.float32)
        tcwv = np.array(tcwv_list, dtype=np.float32)
        emis = np.array(emis_list, dtype=np.float32)
        mu = np.array(mu_list, dtype=np.float32)
        return lst, bt, emis, tcwv, mu

    def simulate(self,
                   selected_type=('tropical', 'mid-latitude', 'polar'),
                   selected_tcwv=[0, 6.3]):
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, _ = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        from pkulast.surface.emissivity import get_band_emissivity
        self.emissivities = get_band_emissivity(self.rsr, self.emis_lib)
        # mtigr = ModtranTIGR946()
        # atm = mtigr.integral(self.rsr)
        matm = RTMAtm(self.rsr)
        tcwv, atm = matm.load(selected_type, selected_tcwv, vza=0)
        TMP = np.array([p.TMP[0] for p in profiles])
        TCWV = np.array([p.TCWV for p in profiles])
        Ta = np.array([p.Ta for p in profiles])
        list_list = np.array(list(map(lambda x: x[0] + x[1], product(self.dTs, TMP))))
        trans_matrix = np.tile(atm["trans"], (len(self.dTs) * len(self.emissivities), 1))
        uward_matrix = np.tile(atm["uward"], (len(self.dTs) * len(self.emissivities), 1))
        dward_matrix = np.tile(atm["dward"], (len(self.dTs) * len(self.emissivities), 1))
        # emiss_matrix = np.tile(emissivities, (len(dTs) * len(profiles), 1))
        emiss_matrix = np.repeat(self.emissivities, len(self.dTs) * len(profiles), axis=0)
        tcwv_matrix = np.squeeze(np.tile(TCWV.reshape(-1, 1), (len(self.dTs) * len(self.emissivities), 1)))
        ta_matrix = np.squeeze(np.tile(Ta.reshape(-1, 1), (len(self.dTs) * len(self.emissivities), 1)))
        lst_maxtrix = np.squeeze(np.tile(list_list.reshape(-1, 1), (len(self.emissivities), 1)))
        brad_matrix = np.tile(np.array(self.rsr.bt2r(list_list)), (len(self.emissivities), 1))
        rad_matrix = (
            trans_matrix * (emiss_matrix * brad_matrix + (1 - emiss_matrix) * dward_matrix)
            + uward_matrix
        )
        mu_maxtrix = np.squeeze(np.tile(np.array(self.mu).reshape(-1, 1), (len(self.dTs) * len(self.emissivities) * len(profiles), 1)))
        bt_matrix = np.array(self.rsr.r2bt(rad_matrix))
        return {
            'lst':lst_maxtrix,
            'rad':rad_matrix,
            'bt':bt_matrix,
            'emis':emiss_matrix,
            'w':tcwv_matrix,
            'mu':mu_maxtrix,
            'tau':trans_matrix,
            'Lu':uward_matrix,
            'Ld':dward_matrix,
            'Ta':ta_matrix,
        }

    def get_indices(self,
                   selected_type=('tropical', 'mid-latitude', 'polar'),
                   selected_tcwv=[0, 6.3]):
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        _, indices = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        if self.emissivities is None:
            from pkulast.surface.emissivity import get_band_emissivity
            self.emissivities = get_band_emissivity(self.rsr, self.emis_lib)
        profile_index = np.zeros(len(ClearSkyTIGR946()), dtype=bool)
        profile_index[indices] = True
        index_matrix = np.squeeze(
            np.tile(profile_index.reshape(-1, 1), (len(self.dTs) * len(self.emissivities), 1))
        )
        return index_matrix

class RTMSimul_old:

    def __init__(
        self,
        rsr,
        emis_lib=DEFAULT_EMIS_LIB,
    ) -> None:
        self.rsr = rsr
        self.emis_lib = emis_lib

    def load(self,
             ts=[-15, 20, 5],
             selected_type=('tropical', 'mid-latitude', 'polar'),
             selected_tcwv=[0, 6.3],
             vza=0):
        from pkulast.surface.emissivity import get_band_emissivity, get_spec_emissivity
        self.d_bound, self.u_bound, self.interval = ts
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, _ = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)

        self.emissivities = get_band_emissivity(self.rsr, self.emis_lib)
        self.spec_emissivities = get_spec_emissivity(self.rsr, self.emis_lib)
        tape5_list = []
        lst_list = []
        for wv, emis in self.spec_emissivities:
            for dt in range(self.d_bound, self.u_bound, self.interval):
                spectra = Spectra(wv, 1 - emis, 100, 'simulated')
                _, tp5 = make_tape5(profiles,
                                    self.rsr,
                                    True,
                                    out_file=None,
                                    flight_altitude=100,
                                    ground_altitude=0.2,
                                    relative_altitude=None,
                                    tbound=None,
                                    dt=dt,
                                    albedo=1.0,
                                    mult=True,
                                    include_solar=False,
                                    vza=vza,
                                    spectra=spectra)
                tape5_list.extend(tp5)
                lst_list.extend([p.TMP[0] + dt for p in profiles])
                break
        radiance = parallel_executor(tape5_list, self.rsr)
        lst = np.array(lst_list, dtype=np.float32)
        return lst, radiance

    def load_cache(self,
                   ts=[-15, 20, 5],
                   selected_type=('tropical', 'mid-latitude', 'polar'),
                   selected_tcwv=[0, 6.3]):
        self.d_bound, self.u_bound, self.interval = ts
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, indices = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        from pkulast.surface.emissivity import get_band_emissivity
        self.emissivities = get_band_emissivity(self.rsr, self.emis_lib)
        mtigr = ModtranTIGR946()
        atm = mtigr.integral(self.rsr)
        rad_list = []
        lst_list = []
        for emis in self.emissivities:
            for dt in range(self.d_bound, self.u_bound, self.interval):
                TMP = np.array([p.TMP[0] + dt for p in profiles])
                rad = atm['trans'] * (emis * self.rsr.bt2r(TMP) +
                                    (1 - emis) * atm['dward']) + atm['uward']
                rad_list.extend(rad)
                lst_list.extend(TMP)
        lst = np.array(lst_list, dtype=np.float32)
        rad = lst = np.array(rad_list, dtype=np.float32)
        return lst, rad

class RTMTrans:

    def __init__(self, rsr) -> None:
        self.rsr = rsr

    def load(self,
             selected_type=('tropical', 'mid-latitude', 'polar'),
             selected_tcwv=[0, 6.5],
             vza=0):
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, indices = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        _, tape5_list = make_tape5_trans(profiles,
                                         self.rsr,
                                         True,
                                         out_file=None,
                                         flight_altitude=None,
                                         ground_altitude=0.2,
                                         relative_altitude=None,
                                         mult=True,
                                         vza=vza)
        transmmittance = parallel_executor(tape5_list, self.rsr, 't')  #
        tcwv = np.array([p.TCWV for p in profiles], dtype=np.float32)
        return tcwv, transmmittance

    def load_cache(self,
                   selected_type=('tropical', 'mid-latitude', 'polar'),
                   selected_tcwv=[0, 6.5]):
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, indices = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        mtigr = ModtranTIGR946()
        transmmittance = mtigr.integral_t(self.rsr)
        tcwv = np.array([p.TCWV for p in profiles], dtype=np.float32)
        return tcwv, transmmittance


class RTMAtm:

    def __init__(self, rsr) -> None:
        self.rsr = rsr

    def load(self,
             selected_type=('tropical', 'mid-latitude', 'polar'),
             selected_tcwv=[0, 6.5],
             vza=0):
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, indices = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        _, tape5_list = make_tape5_atm(profiles,
                                         self.rsr,
                                         True,
                                         out_file=None,
                                         flight_altitude=None,
                                         ground_altitude=0.2,
                                         relative_altitude=None,
                                         mult=True,
                                         vza=vza)
        self.uward = parallel_executor(tape5_list[0], self.rsr, mode='r')
        self.dward = parallel_executor(tape5_list[1], self.rsr, mode='r')
        self.trans = parallel_executor(tape5_list[2], self.rsr, mode='t')
        tcwv = np.array([p.TCWV for p in profiles], dtype=np.float32)
        result = {
            'uward': self.uward,
            'dward': self.dward,
            'trans': self.trans
        }
        return tcwv, result

    def load_cache(self,
                   selected_type=('tropical', 'mid-latitude', 'polar'),
                   selected_tcwv=[0, 6.5]):
        from pkulast.atmosphere.profile import ClearSkyTIGR946
        profiles, indices = ClearSkyTIGR946(selected_type,
                                            selected_tcwv,
                                            ref_indices=True)
        mtigr = ModtranTIGR946()
        atm = mtigr.integral(self.rsr)
        tcwv = np.array([p.TCWV for p in profiles], dtype=np.float32)
        return tcwv, atm


###############################################################################
# RTTOV simulation
###############################################################################

# RTTOV implementation

RTTOV_DIR = RTM_RTTOV13_DIR # TODO, fixed location, need to be modified
import os
import sys
import glob
import time
import numpy as np
# import rioxarray
from collections.abc import Iterable

pyrttov_dir = os.path.join(RTTOV_DIR,'wrapper')
if not pyrttov_dir in sys.path:
    sys.path.append(pyrttov_dir)
try:
    import pyrttov
except:
    pass

from pkulast.utils.spectrum import convert_spectral_density, convert_spectral_domain
from pkulast.utils.physics import thermodynamics
from pkulast.utils.physics.atmosphere import relative_humidity2vmr, vmr2relative_humidity

def relative_humidity2specific_humidity(p):
    """RH: 1-100; P: hPa; TMP: K
    specific_humidity: g/kg
    """
    RH, P, TMP = p.RH, p.P, p.TMP
    vmr = relative_humidity2vmr(RH * 1e-2, P * 1e2, TMP)
    return thermodynamics.vmr2specific_humidity(vmr)

def specific_humidity2relative_humidity(Q, P, T):
    vmr = thermodynamics.specific_humidity2vmr(Q * 1e-6)
    print(vmr)
    return vmr2relative_humidity(vmr, P * 1e2, T)

def relative_humidity2ppmv(p):
    """RH: 1-100; P: hPa; TMP: K
    ppmv: ppmv
    """
    RH, P, TMP = p.RH, p.P, p.TMP
    vmr = relative_humidity2vmr(RH * 1e-2, P * 1e2, TMP)
    return vmr * 1e6

class RttovSensor:

    def __init__(self, rttov_dir=RTTOV_DIR):
        self.rtcoef_dir = os.path.join(rttov_dir, f"rtcoef_rttov13/rttov13pred54L")
        self.sccldcoef_dir = os.path.join(rttov_dir, f"rtcoef_rttov13/cldaer_visir")
        self.sccldcoef_ironly_dir = os.path.join(rttov_dir, f"rtcoef_rttov13/cldaer_ir")
        self.mfasiscld_dir = os.path.join(rttov_dir, f"rtcoef_rttov13/mfasis_lut")
        self.coefs_path = glob.glob(os.path.join(self.rtcoef_dir, "rtcoef_*.dat"))
        self.hires_coefs_path = glob.glob(os.path.join(self.rtcoef_dir, "rtcoef_*.H5"))

    @property
    def sensors(self):
        # only for non-hi-res sensors
        res = []
        for filename in self.coefs_path:
            longname = os.path.splitext(os.path.basename(filename))[0]
            info = longname.split('_')
            res.append("_".join(info[1:4]))
        return set(res)

    def search(self, keyword, hires=False):
        if hires:
            return [name for name in self.hires_sensors if keyword in name]
        else:
            return [name for name in self.sensors if keyword in name]

    def load_sensor(self, sensor, gas="", ironly=False):
        addstr = "_{}".format(gas) if gas else ""
        addstr += "" if not ironly else "_ironly"
        self.name = "{}{}".format(sensor, addstr)
        self.ironly = ironly
        self.gas = gas
        assert len(self.get_rtcoef()) > 0

    def get_rtcoef(self):
        """get sensor coef path.
        Args:
            gas: "", "o3co2", "o3", "7gas"
            inonly: True, False
        """
        # ext = ".H5" if hires else '.dat'

        return glob.glob(os.path.join(self.rtcoef_dir, f"rtcoef_{self.name}*.dat"))

    def get_sccldcoef(self):
        """Aerosol and cloud scattering coefficients
        """
        sccldcoef_dir = self.sccldcoef_ironly_dir if self.ironly else self.sccldcoef_dir
        return glob.glob(os.path.join(sccldcoef_dir, f"sccldcoef_{self.name}*.dat"))

    def get_mfasis_cld(self):
        """MFASIS fast visible/near-IR scattering model
        """
        return glob.glob(os.path.join(self.mfasiscld_dir, f"rttov_mfasis_cld_{self.name}*.dat"))

    @property
    def hires_sensors(self):
        res = []
        for filename in self.hires_coefs_path:
            longname = os.path.splitext(os.path.basename(filename))[0]
            info = longname.split('_')
            res.append("_".join(info[1:4]))
        return set(res)


def get_brdf(atlas_path, months, rttov_class):
    """ used < 5um for solar affected bands
    """
    brdf_atlas = pyrttov.Atlas()
    brdf_atlas.AtlasPath = atlas_path
    for month in set(months):
        brdf_atlas.loadBrdfAtlas(month, rttov_class) # supply Rttov object to enable single-instrument initialisation
    brdf_atlas.IncSea = False # do not use BRDF atlas for sea surface types

    return brdf_atlas


def get_iremis(atlas_path, months):
    """UWIRemis(0.1), CAMEL 2007(0.05), and CAMEL climatology IR emissivity atlases(0.05)
    include a correction for zenith angle effects(needs) vza, sza(<85 at day and >85 at night),
    PC-RTTOV land/sea coefficients were trained using the UWIRemis atlas, which is
    recommended to use in PC-RTTOV
    """
    ir_atlas = pyrttov.Atlas()
    ir_atlas.AtlasPath = atlas_path
    for month in set(months):
        ir_atlas.loadIrEmisAtlas(month, ang_corr=True)# include angular correction, but do not initialise for single-instrument
    return ir_atlas

def get_mwemis(atlas_path, month):
    """ TELSEM2 MW atlas and interpolator(0.25), CNRM MW atlas(2014, 2015)
    frequencies(<19GHz and >85GHz), return fixed value in 19GHz and 85 GHz
    a dynamic value between 19 GHz and 85 GHz.
    """
    mw_atlas = pyrttov.Atlas()
    mw_atlas.AtlasPath = atlas_path
    mw_atlas.loadMwEmisAtlas(month)

    return mw_atlas


def expand2nprofiles(val, nprofiles):
    """Transform 1D array to a [nprofiles, nlevels] array"""
    if isinstance(val, Iterable):
        result = np.empty((nprofiles, len(val)), dtype=val.dtype)
        for i in range(nprofiles):
            result[i, :] = val[:]
    else:
        result = np.asarray([val, ] * nprofiles)
    return result

class RttovWrapper:

    def __init__(self, rttov_dir=RTTOV_DIR):
        self.rttov_dir = rttov_dir
        self.rtcoef_dir = os.path.join(rttov_dir, f"rtcoef_rttov13/rttov13pred54L")
        self.emis_dir = os.path.join(rttov_dir, "emis_data")
        self.brdf_dir = os.path.join(rttov_dir, "brdf_data")
        self.rttov_ = pyrttov.Rttov()

    def set_sensor(self, sensor, channels):
        self.sensor = sensor
        self.rttov_.FileCoef = sensor.get_rtcoef()[0]
        # For MFASIS LUT
        # self.rttov_.FileMfasisCld = sensor.get_mfasis_cld()[0]

        # For cloud/aerosol coef
        # self.rttov_.FileSccld  = sensor.get_sccldcoef()[0]

        self.channels = channels
        self.nchannels = len(channels)
        try:
            self.rttov_.loadInst()
            self.selected_bands = np.asarray(channels, dtype=int)
            band_str = ' '.join(map(str, channels))
            self.set_options()
            print(f"Loading instruments {self.rttov_.InstId}, contains {self.rttov_.Nchannels} channels\nSelecting bands {band_str}") # InstId Nchannels
        except pyrttov.RttovError as e:
            sys.stderr.write("Error loading instrument(s): {!s}".format(e))
            sys.exit(1)

    def set_aerosol(self):
        raise NotImplementedError

    def set_clouds(self):
        raise NotImplementedError

    def update_options(self, options):
        self.rttov_.updateOptions(options)

    def set_options(self, mode="IR"):
        self.rttov_.Options.AddAerosl = True
        self.rttov_.Options.StoreTrans = True
        self.rttov_.Options.StoreRad = True
        self.rttov_.Options.StoreEmisTerms = True
        self.rttov_.Options.StoreRad2 = True
        self.rttov_.Options.Nthreads = 10
        self.rttov_.Options.NprofsPerCall = 50
        self.rttov_.Options.DoCheckinput = False
        self.rttov_.Options.Verbose = False
        self.rttov_.Options.VerboseWrapper = False
        self.rttov_.Options.AddInterp = True
        self.rttov_.Options.AddSolar = True # important
        # self.rttov_.Options.AddClouds = True

        # Input profiles can be clipped to the regression limits when the limits are exceeded
        self.rttov_.Options.ApplyRegLimits = True
        # self.rttov_.Options.GridBoxAvgCloud = True
        # self.rttov_.Options.UserCldOptParam = False

        #  self.rttov_.Options.CO2Data = True
        #  self.rttov_.Options.OzoneData = True
        # N2OData, COData, CH4Data, SO2Data
        # ClwData for MW Clear Sky only

        # Scatt Model
        # IrScattModel, [1, 2]. 1 for DOM, 2 for Chou-scaling(default)
        # VisScattModel, [1, 2, 3]. 1 for DOM (default), 2 for single scattering, 3 for MFASIS(cloud only)

        # 1.DOM, both for VIS and IR, specularity data not used, slow
        # 2.Chou-scaling, multiple-scatting for TIR emission term only
        # 3.Single-scattering, for solar source term only
        # 4.MFASIS, for solar source term only
        if mode == 'IR':
            self.rttov_.Options.VisScattModel = 1  # for IR sim necessary!
            self.rttov_.Options.IrScattModel  = 2
            # IrSeaEmisModel
        elif mode == "VIS":
            self.rttov_.Options.VisScattModel = 3 # MFASIS=3
            self.rttov_.Options.IrScattModel  = 2
        else:
            pass


    def set_params(self,
                   profiles,
                   tbound=300,
                   emis=None,
                   brdf=None,
                   diffuse=None,
                   vza=0,
                   vaa=0,
                   elev=None,
                   sza=None,
                   saa=None,
                   surftype=0,
                   watertype=0):
        nlevels = profiles[0].layer_number
        self.nprofiles = nprofiles = len(profiles)
        # H km
        # P hPa
        # O3 g/kg
        # RH %
        # TMP K
        # Ts K
        rttov_profiles = pyrttov.Profiles(nprofiles, nlevels)

        # gas_unit, 1
        # 2 => ppmv over moist air
        # 1 => kg/kg over moist air (the default)
        # 0 => ppmv over dry air
        rttov_profiles.GasUnits = 2 # [ ppmv_dry=0, kg_per_kg=1, ppmv_wet=2 ]
        rttov_profiles.MmrCldAer = True # kg/kg for clouds and aerosoles

        # Gas, [nprofiles, nlevels]
        rttov_profiles.P = np.asarray([p.P[::-1] for p in profiles])
        rttov_profiles.T = np.asarray([p.TMP[::-1] for p in profiles])

        Qv = np.asarray([relative_humidity2ppmv(p)[::-1] for p in profiles])
        # avoid RTTOV complaining about too low Qv
        Qv[Qv<1e-9] = 1e-9
        rttov_profiles.Q = Qv

        # rttov_profiles.CO2 = np.ones((nprofiles,nlevels))*3.743610E+02
        # rttov_profiles.O3 = None
        # rttov_profiles.N2O = None
        # rttov_profiles.CO = None
        # rttov_profiles.SO2 = None
        # rttov_profiles.CH4 = None


        # angles, [nprofiles, 4], vza, vaa, sza, saa
        SZA = expand2nprofiles(sza, nprofiles)
        SAA = expand2nprofiles(saa, nprofiles)
        if sza is None or sza is None:
            for i, p in enumerate(profiles):
                acq_time = pytz.utc.localize(p.acq_time)
                z, a = get_sun_position(p.latitude, p.longitude,
                                        acq_time)
                if sza is None:
                    SZA[i] = z
                if saa is None:
                    SAA[i] = a
            # sza = np.asarray([p.SZA for p in profiles])
        rttov_profiles.Angles = np.c_[expand2nprofiles(vza, nprofiles),
                                      expand2nprofiles(vaa, nprofiles),
                                      SZA,
                                      SAA]

        # s2m, [nprofiles, 6], p2m, t2m, q2m, u10m, v10m, wfetch
        # surface 2m pressure, hPa
        # surface 2m temperature, K
        # surface 2m wv(op), gas_unit
        # wind 10m u(op), m/s
        # wind 10m v(op), m/s
        # wind fetch(op) 100000m

        p2m = np.asarray([p.P[0] for p in profiles])
        t2m = np.asarray([p.TMP[0] for p in profiles])
        q2m = np.asarray([relative_humidity2ppmv(p) for p in profiles])[:, 0]
        rttov_profiles.S2m = np.c_[p2m,
                                   t2m,
                                   q2m,
                                   expand2nprofiles(0, nprofiles),
                                   expand2nprofiles(0, nprofiles),
                                   expand2nprofiles(10000, nprofiles)]

        # skin [nprofiles, 9], ts, salinity, snow_frac, foam_frac, fastem_coef*5
        # skin temperature, K
        # salinity(op), for water surfaces (PSU), typical value for sea water: 35
        # snow cover fraction(op), 0-1, 0
        # ocean foam fraction(op), 0-1, 0
        # FASTEM coefs(op), 1-5 , [3.0, 5.0, 15.0, 0.1, 0.3]
        rttov_profiles.Skin = np.c_[expand2nprofiles(tbound, nprofiles),
                                    expand2nprofiles(0, nprofiles),
                                    expand2nprofiles(0, nprofiles),
                                    expand2nprofiles(0, nprofiles),
                                    expand2nprofiles(3, nprofiles),
                                    expand2nprofiles(5, nprofiles),
                                    expand2nprofiles(15, nprofiles),
                                    expand2nprofiles(0.1, nprofiles),
                                   expand2nprofiles(0.3, nprofiles)]

        # surftype, [nprofiles, 2], surftype, watertype
        # surface type, 0=land, 1=sea, 2=sea-ice
        # water type(op),for water surfaces: 0=fresh water, 1=ocean
        rttov_profiles.SurfType = np.c_[expand2nprofiles(surftype, nprofiles),
                                        expand2nprofiles(watertype, nprofiles)]

        lats = np.asarray([p.latitude for p in profiles]) # -90-90
        lons = (np.asarray([p.longitude for p in profiles]) + 360) % 360 # op, 0-360

        if elev is None:
            elevs = np.asarray([get_elevation(*loc) for loc in zip(lats, lons)])# km
        else:
            elevs = expand2nprofiles(elev, nprofiles)
        # surfgeom, [nprofiles, 3], lat degree, lon degree, elev m
        rttov_profiles.SurfGeom = np.c_[lats, lons, elevs]

        years = np.asarray([p.acq_time.year for p in profiles])
        months = np.asarray([p.acq_time.month for p in profiles])
        days = np.asarray([p.acq_time.day for p in profiles])
        hours = np.asarray([p.acq_time.hour for p in profiles])
        minutes = np.asarray([p.acq_time.minute for p in profiles])
        seconds = np.asarray([p.acq_time.second for p in profiles])
        # datetimes, [nprofiles, 6], year, month, day, hour, minute, second
        rttov_profiles.DateTimes = np.c_[years, months, days, hours, minutes, seconds]

        #Optional
        # SimpleCloud [nprofiles, 2], ctp, cfraction
        # ctp: Cloud top pressure for simple cloud
        # cfraction: Cloud fraction for simple cloud
        # ClwScheme, [nprofiles, 2], visible/ir (clw_scheme, clwde_param)
        # IceCloud, [nprofiles, 2], ice_scheme, icede_param
        # Zeeman, [nprofiles, 2], Be, cosbk,
        # Be: Earth magnetic field strength(Gauss)
        # cosbk: Cosine of the angle between the Earth magnetic field and wave propagation direction
        # Cfrac, cfrac, cloud fraction, 0-1
        # Cirr, ciw, cloud ice water

        # eaxmple:
        # Inso, aer_inso, insoluble aerosol

        # For MW channels
        # rttov_profiles.CLW = clw
        # rttov_profiles.Cfrac = cfrac  # cloud fraction

        # WATER CLOUDS
        # dummy = np.ones((nprofiles, 2))
        # dummy[:,0] = 2  # clw_scheme : (1) OPAC or (2) Deff scheme
        # dummy[:,1] = 1  # clwde_param : currently only "1" possible
        # rttov_profiles.ClwScheme = dummy
        # rttov_profiles.Clwde = 20*np.ones((nprofiles,nlevels))

        # Cloud types - concentrations in kg/kg
        # clw  [nprofiles, nlevels]# cloud liquid water mixing ratio kg kg-1
        # ciw  [nprofiles, nlevels]# cloud ice water  mixing ratio kg kg-1
        # rttov_profiles.Stco = clw  # Stratus Continental STCO
        # rttov_profiles.Stma = 0*clw  # Stratus Maritime STMA
        # rttov_profiles.Cucc = 0*clw  # Cumulus Continental Clean CUCC
        # rttov_profiles.Cucp = 0*clw  # Cumulus Continental Polluted CUCP
        # rttov_profiles.Cuma = 0*clw  # Cumulus Maritime CUMA
        # rttov_profiles.Cirr = ciw  # all ice clouds CIRR

        # Ice Cloud [2][nprofiles]: ice_scheme, idg
        # icecloud = np.array([[1, 1]], dtype=np.int32)
        # rttov_profiles.IceCloud = expand2nprofiles(icecloud, nprofiles)
        # rttov_profiles.Icede = 60 * np.ones((nprofiles,nlevels))  # microns effective diameter

        self.rttov_.Profiles = rttov_profiles

        # surfemisrefl [4, nprofiles, nchannels] in/out
        # emissivity, BRDF, diffuse reflectance, specularity
        surfemisrefl = np.zeros((4, self.nprofiles, self.nchannels))
        # 1.surface emissivities, for IR/MW bands
        if emis is None:# obtain emissivity from database
            try:
                iremis_atlas = get_iremis(self.emis_dir, months)
                surfemisrefl[0, :, :] = iremis_atlas.getEmisBrdf(self.rttov_)[:, self.selected_bands - 1]
            except pyrttov.RttovError as e:
                sys.stderr.write("Error calling atlas: {!s}".format(e))
        else:
            if not isinstance(emis, Iterable):
                emis = np.ones(self.nchannels) * emis
            surfemisrefl[0, :, :] = expand2nprofiles(emis, self.nprofiles)
        # 2.BRDFs(reflectance), for solar-affected bands
        if brdf is None:
            try:
                brdf_atlas = get_brdf(self.brdf_dir, months, self.rttov_)
                surfemisrefl[1, :, :] = brdf_atlas.getEmisBrdf(self.rttov_)[:, self.selected_bands - 1]
            except pyrttov.RttovError as e:
                sys.stderr.write("Error calling atlas: {!s}".format(e))
        else:
            if not isinstance(brdf, Iterable):
                brdf = np.ones(self.nchannels) * brdf
            surfemisrefl[1, :, :] = expand2nprofiles(brdf, self.nprofiles)
            # surfemisrefl[1, :, :] = expand2nprofiles(albedo / np.pi, self.nprofiles)

        # 3.diffuse reflectance, for VIS/NIR bands
        if diffuse is not None:
            if not isinstance(diffuse, Iterable):
                diffuse = np.ones(self.nchannels) * diffuse
            surfemisrefl[2, :, :] = expand2nprofiles(diffuse, self.nprofiles)
        else:
            surfemisrefl[2, :, :] = 0

        # 4.specularity
        surfemisrefl[3, :, :] = 0

        # if sea, select sea emissivity
        seaflag = self.rttov_.Profiles.SurfType[:,0] == 1
        surfemisrefl[:, seaflag, :] = -1
        self.rttov_.SurfEmisRefl = surfemisrefl

    def convert_radiance(self, radiance):
        wn = self.rttov_.WaveNumbers[self.selected_bands - 1] # cm^-1
        if len(radiance.shape) == 1:
            return convert_spectral_density(wn, radiance, 'nl')[1] * 1e-3
        else:
            return np.squeeze([convert_spectral_density(wn, rad, 'nl')[1] for rad in radiance]) * 1e-3

    def run(self,):
        try:
            start = time.time()
            self.rttov_.runDirect(self.channels)
            print('runDirect cost %.2fs' %(time.time() - start))
        except pyrttov.RttovError as e:
            sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
            sys.exit(1)
        if self.rttov_.RadQuality is not None:
            print('Quality (qualityflag>0), #issues:', np.sum(self.rttov_.RadQuality > 0))

        surfemisrefl = self.rttov_.SurfEmisRefl
        radiances_wn = self.rttov_.Rads # mW/sr/m^2/cm^-1
        wn = self.rttov_.WaveNumbers[self.selected_bands - 1] # cm^-1
        wl = convert_spectral_domain(wn, 'nl' ) # um
        result = {
            'uward':self.rttov_.Rad2UpClear,
            'dward':self.rttov_.Rad2DnClear,
            'trans':self.rttov_.TauTotal,
            'sun_tau1':self.rttov_.TauSunTotalPath1,
            'sun_tau2':self.rttov_.TauSunTotalPath2,
        #             'height':self.rttov_.GeometricHeight,
            'rads':radiances_wn,
            'bt':self.rttov_.BtRefl,
            'refl':self.rttov_.Refl,
            'emis':surfemisrefl[0],
            'brdf':surfemisrefl[1],
            'surfemisrefl':surfemisrefl,
            'wn':wn,
            'wl':wl,
        }
        return result

    def searchOptions(self, keyword):
        for item in dir(self.rttov_.Options):
            if keyword in item.lower():
                print(item)

    def printSummary(self,):
        self.rttov_.printOptions()
        self.rttov_.printSurfEmisRefl()
        # for item in dir(rw.rttov_):
        # if item[0] != '_' and item[-1] != 'K':
        #     print(item)

    def runK(self,):#Jacobian
        try:
            start = time.time()
            self.rttov_.runK(self.channels)
            print('runDirect cost %.2fs' %(time.time() - start))
        except pyrttov.RttovError as e:
            sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
            sys.exit(1)

################################################################################
# CRTM Interface
################################################################################

CRTM_DIR = os.path.join(CFG.get('extra'), 'rtm/CRTM')
class CrtmWrapper:

    def __init__(self, rsr=CRTM_DIR):
        self.crtm_dir = CRTM_DIR