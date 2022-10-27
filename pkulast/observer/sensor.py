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
Sensor Specification.
=====================

Provides sensor specifications.

"""

import re
import logging
import numpy as np
from math import floor, sqrt, exp, log
from binascii import unhexlify
from collections import defaultdict, abc
from pylab import mpl
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.linear_model import LinearRegression

from pkulast.config import *
from pkulast.constants import *
from pkulast.utils.collections import convert2str
from pkulast.utils.spectrum import get_central_wave, convert2wavenumber, get_effective_quantity
from pkulast.utils.thermal import convert_wave_unit, planck_wl


LOG = logging.getLogger(__name__)
mpl.rcParams['figure.figsize'] = (7, 4.5)
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# mpl.rcParams['font.size'] = 24
rc_config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "font.family": ["Times New Roman"],
    "font.serif": ["SimSun"],
}
plt.rcParams.update(rc_config)

def get_cmap(n, name='hsv'):
    ''' Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
  RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class Band(object):
    '''Band filter function
    '''
    def __init__(self, x, y, sensor, band_number, center=None, fwhm=None, unit='nm', desc=None):
        if len(x) != len(y):
            raise ValueError('ValueError: x, y should have a same length')
        if not np.array_equal(sorted(x), x):
            x = np.array(list(reversed(x)), dtype=np.float64)
            y = np.array(list(reversed(y)), dtype=np.float64)
        x, indices = np.unique(x, return_index=True)
        y = y[indices]
        self.wv = x # wavelength
        self.response = y  / np.max(y)# response
        self.wv = self.wv[self.response > 1e-3]
        self.response = self.response[self.response > 1e-3]
        self.sensor = sensor
        self.band_number = band_number
        self.center = center if center else self.wv[np.argmax(self.response)]
        roots = UnivariateSpline(self.wv, self.response - np.max(self.response) / 2, s=0).roots()
        self.l, self.r = roots[0], roots[-1]
        self.fwhm = fwhm if fwhm else abs(self.l - self.r)
        self.unit = unit
        self.desc = desc if desc else f'sensor {sensor}, band {band_number}'
        self.generate_lut()

    def plot(self, color='r', show_span=True, subplot=False, label=None, unit=None):
        '''Plot band filter
        '''
        if show_span:
            plt.axvspan(self.l, self.r, facecolor='g', alpha=0.3)
        if label is None:
            label = f'Band{self.band_number}'
        if unit is not None:
            wv = self.get_wv(unit)
        else:
            wv = self.wv
        plt.plot(wv, self.response, '-', color=color, label=f'{label}', lw=0.8)
        if not subplot:
            plt.xlabel(f'WaveLength({self.unit})')
            plt.ylabel('Spectral Response')
            plt.legend(loc="upper right")
            plt.ylim(0, 1.05)
            plt.xlim(*self.xlim)
            plt.title(f' "{self.sensor}" WL-SR Relationship')
            plt.show()
            plt.close()

    def plot_span(self, axis, color='g', alpha=0.3):
        axis.axvspan(self.l, self.r, alpha=alpha) #facecolor=color,

    def plot_response(self, axis, color='g'):
        axis.plot(self.wv, self.response, '-', label=f'Band{self.band_number}', lw=1.2)

    def bt2r(self, bt):
        ''' Convert bright temperature to radiance
        '''
        radiances = planck_wl(self.wv, bt, self.unit)
        if np.isscalar(bt):
            return np.sum(radiances * self.response) / np.sum(self.response)
        return np.sum(radiances * self.response, axis=1) / np.sum(self.response)

    def r2bt(self, radiance):
        """ Convert radiance to bright temperature
        """
        # assert np.isscalar(radiance)
        if not self._check_boundary(radiance):
            return -999
        lines, _ = self.lut.shape
        l, r = self._bi_search(0, lines-1, radiance)
        if l == r:
            return self.lut[l][1]
        else:
            return float(interp1d(self.lut[l:r+1, 0], self.lut[l:r+1, 1])(radiance))

    def get_wv(self, unit='um'):
        wv = self.wv
        if unit == 'um':
            if self.unit == 'nm':
                wv = wv * 1e-3
        elif unit == 'nm':
            if self.unit == 'um':
                wv = wv * 1e3
        else:
            raise ValueError(f'unsupported unit: {unit}!')
        return wv

    def convert_wv(self, wv, unit='um'):
        if unit == 'um':
            if self.unit == 'nm':
                wv = wv * 1e-3
        elif unit == 'nm':
            if self.unit == 'um':
                wv = wv * 1e3
        else:
            raise ValueError(f'unsupported unit: {unit}!')
        return wv

    def get_unit(self):
        if self.unit == 'nm':
            return 'Nanometer'
        elif self.unit == 'um':
            return 'Micrometer'
        else:
            raise ValueError(f'unsupported unit: {self.unit}!')

    def get_lut(self):
        return self.lut[:, 1]

    def generate_lut(self):
        BTs = np.linspace(MIN_TMP, MAX_TMP, MAX_STEPS)
        # self.radiances = list(map(lambda bt:self.bt2r(bt), BTs))
        self.radiances = self.bt2r(BTs)
        self.lut = np.vstack((self.radiances, BTs)).transpose()

    def _bi_search(self, l, r, val):
        if r - l <= 1:
            return l, r
        mid = int((r + l) / 2)
        mid_val = self.lut[mid][0]
        if mid_val > val:
            return self._bi_search(l, mid, val)
        elif mid_val < val:
            return self._bi_search(mid, r, val)
        else:
            return mid, mid

    def _check_boundary(self, val):
        if self.lut[0][0] > val or self.lut[-1][0] < val:
            print(f"{self.sensor}-{self.band_number} {val}")
            LOG.error("radiance value out of the range of lookup-table!")
            return False
        return True

    def interp(self, spectral_wv, spectral_res, unit='nm'):
        if unit == 'um':
            if self.unit == 'nm':
                spectral_wv = spectral_wv * 1e3
        elif unit == 'nm':
            if self.unit == 'um':
                spectral_wv = spectral_wv * 1e-3
        else:
            raise ValueError(f'unsupported unit: {unit}!')
        return np.sum(interp1d(spectral_wv, spectral_res, fill_value="extrapolate")(self.wv) * self.response) / np.sum(self.response)

    def effective_value(self, spectral_wv, spectral_res, unit='nm'):
        if unit == 'um':
            if self.unit == 'nm':
                spectral_wv = spectral_wv * 1e3
        elif unit == 'nm':
            if self.unit == 'um':
                spectral_wv = spectral_wv * 1e-3
        else:
            raise ValueError(f'unsupported unit: {unit}!')
        interp_response = interp1d(spectral_wv, spectral_res, fill_value="extrapolate")(self.wv)
        return get_effective_quantity(self.wv, interp_response, self.response)

    def effective_lambda(self):
        """ Effective wavelength
        Returns:
            float: effective wavelength in mircometers
        """
        scale_factor = 1
        if self.unit == 'nm':
            scale_factor = 1e-3
        elif self.unit == 'um':
            scale_factor = 1
        else:
            raise ValueError(f'unsupported unit: {self.unit}!')
        return np.sum(self.wv * self.response) / np.sum(self.response) * scale_factor

    def b_lambda(self):
        return 1.43877e4 / self.effective_lambda()

    def derivation_ab(self, min_t=0- ZERO_TEMPERATURE, max_t=60- ZERO_TEMPERATURE):

        BTs = np.linspace(min_t, max_t, MAX_STEPS)
        # self.radiances = list(map(lambda bt:self.bt2r(bt), BTs))
        radiances = self.bt2r(BTs)
        model = LinearRegression()
        model.fit(BTs.reshape(-1, 1), radiances)
        return model.intercept_, model.coef_[0]


    @property
    def xlim(self):
        l = self.l - self.fwhm * EXTENT_FACTOR
        r = self.r + self.fwhm * EXTENT_FACTOR
        l = max(l, min(self.wv))
        r = min(r, max(self.wv))
        return l if l > 0 else 0, r

    def __repr__(self):
        return f"{self.get_unit()} filter data for {self.desc}; created by simpir.\n"

    def __str__(self):
        '''Band string
        '''
        band_str = 'B{}  CENTER:        {:>10.4f}{}   FWHM:      {:>10.4f} {}\n'.format(self.band_number, self.center, self.unit.upper(), self.fwhm, self.unit.upper())
        for i, wv in enumerate(self.wv):
            band_str += "{:>10.4f}{:>10.7f}\n".format(wv, self.response[i])
        return band_str


def get_unit_symbol(unit):
    if unit == 'nm':
        return '$\mathrm{nm}$'
    elif unit == 'um':
        return '$\mathrm{\mu m}$'
    else:
        raise ValueError(f'unsupported unit: {unit}!')


class RelativeSpectralResponse(object):
    """Container for the relative spectral response functions for various satellite imagers."""
    @classmethod
    def parse(cls, filename, sensor=None):
        ''' Read spectral response filters to array
        '''
        if not sensor:
            sensor = os.path.splitext(os.path.basename(filename))[0]
        bands = []
        unit = None
        center = 0
        fwhm   = 0
        with open(filename, "r") as f:
            inline = False # recording
            sig_band = []
            for line in f.readlines():
                if not unit and 'Micrometer' in line:
                    unit = 'um'
                elif not unit and 'Nanometer' in line:
                    unit = 'nm'
                line = line.replace('CENTER', 'Center')
                if line.find("Center") != -1:
                    info = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
                    center = float(info[0])
                    fwhm = None if len(info) == 1 else float(info[1])
                    if sig_band:
                        band_number = len(bands) + 1
                        data = np.asarray(sig_band)
                        bands.append(Band(data[:, 0], data[:, 1], sensor, band_number, center, fwhm, unit))
                    inline = True
                    sig_band = []
                    continue
                elif not inline:
                    continue
                else:
                    sig_band.append(list(map(lambda x:float(x), line.strip().split())))
            band_number = len(bands) + 1
            data = np.asarray(sig_band)
            bands.append(Band(data[:, 0], data[:, 1], sensor, band_number, center, fwhm, unit))
        return RelativeSpectralResponse(sensor, bands=bands)

    @classmethod
    def simulate(cls, sensor, centers, fwhms, unit='nm'):
        ''' Simulate filter for HyperSpectral Sensor(assumes Gaussian with maximum response of 1.0).
        '''
        band_count = len(centers)
        assert len(centers) == len(fwhms)
        bands = []
        for i, center_wave in enumerate(centers):
            if unit == 'nm':
                CENTER = center_wave
                FWHM = fwhms[i]
            elif unit == 'um':
                CENTER = center_wave * 1e3
                FWHM = fwhms[i] * 1e3
            else:
                raise ValueError(f'unsupported units: {unit}, use mircometer or nanometer instead')
            step = 3
            count = floor(FWHM / step + 0.5) + 10 # sampling count
            wave_length = np.zeros(count * 2 + 1)
            response = np.zeros(count * 2 + 1)
            for index in range(2 * count + 1):
                wave_length[index] = CENTER + (index - count) * step
                val = (index - count) * step
                C = val / FWHM * 2 * sqrt(log(2))
                response[index] = exp(- 1 * C ** 2)
            bands.append(Band(wave_length, response, sensor, i+1, CENTER, FWHM, "nm"))
        return RelativeSpectralResponse(sensor, bands=bands)

    @classmethod
    def composite_plot(cls, *args, show_span=True, loc='outside', unit="um", **kwargs):
        """Plot the composite response function of the given relative spectral response functions.
        Args:
            *args: RelativeSpectralResponse objects.
            show_span: Whether to show the span of the response function.
            loc: Location of the legend.
            unit: Unit of the wavelength. Options: ['um', 'nm']
        Returns:
            None
        """
        total_band_count = sum([rsr.selected_band_count for rsr in args])
        total_cmap = get_cmap(total_band_count).reversed()
        left_xlim = []
        right_xlim = []
        title = ""
        for rsr in args:
            title += f"{rsr.sensor} "
            for iband in rsr.selected_bands:
                title += f"B{iband} "
                rsr.bands[iband - 1].plot(total_cmap(iband - 1), show_span=show_span, subplot=True, label=f"{rsr.sensor} Band{iband}", unit=unit)
                l, r = rsr.bands[iband-1].convert_wv(np.asarray(rsr.bands[iband-1].xlim), unit)
                left_xlim.append(l)
                right_xlim.append(r)
        plt.xlabel(f'WaveLength({get_unit_symbol(unit)})')
        plt.ylabel('Spectral Response')
        if loc == 'outside':
            plt.legend(scatterpoints=1, labelspacing=0.4, columnspacing=2, markerscale=2, bbox_to_anchor=(1.01, 1.02), ncol=2, **kwargs)
        else:
            plt.legend(loc=loc)
        plt.ylim(0, 1.05)
        plt.xlim(*(min(left_xlim), max(right_xlim)))
        plt.title(f' "{title}" ')
        plt.show()
        plt.close()

    def __init__(self, sensor, **kwargs):
        """ create the instance either sensor or from filename and load the data.
        """
        self.rsr = {}
        self.description = "Unknown"
        self.bands = []
        self.unit = '1e-6 m'
        self.si_scale = 1e-6
        self._wavespace = WAVE_LENGTH
        self._source = 'USER'
        self.band_count = 0
        self.sensor = sensor
        self.band_names = None
        if 'bands' in kwargs:
            self.bands = kwargs['bands']
            self.band_count = len(self.bands)
        else:
            if self.sensor in RSR_LIB:
                self._check_sensor()
                self.load()
            else:
                raise ValueError(f'{self.sensor} is not included in sensor library!')
        self.cmap = get_cmap(self.band_count)
        self.rev_camp = self.cmap.reversed()
        self.selected_bands = range(1, self.band_count + 1)
        self.unit = self.bands[0].unit
        if 'name' in kwargs:
            self.sensor = kwargs['name']

    def _check_sensor(self):
        """Check and try fix sensor name if needed."""
        if self.sensor in ENVI_RSR_LIB:
            self._source = 'ENVI'
        elif self.sensor in NWP_RSR_LIB:
            self._source = 'NWP'
        elif self.sensor in HDF5_RSR_LIB:
            self._source = 'HDF5'

    def load(self):
        """Read the internally format relative spectral response data."""

        # ENVI source
        if self._source == 'ENVI':
            # read hdr
            with open(f'{ENVI_RSR_DIR}/{self.sensor}.hdr', 'r', encoding='utf-8') as f:
                hdr = defaultdict()
                string = ''.join(f.readlines())
                samples = re.search(r'samples\s+=\s+(?P<samples>\d+)', string)
                lines = re.search(r'lines\s+=\s+(?P<lines>\d+)', string)
                data_type = re.search(r'data type\s+=\s+(?P<data_type>\d+)', string)
                units = re.search(r'wavelength units\s+=\s+(?P<units>[^\n]+)', string)
                # factor = re.search(r'reflectance scale factor\s+=\s+(?P<factor>[-+]?[0-9]*\.?[0-9]+)', string)
                spectra_names = re.search(r'spectra names\s+=\s+{(?P<spectra_names>[^}]+)}', string)
                wavelength = re.search(r'wavelength\s*=\s*{(?P<wavelength>[^}]+)}', string)
                if not (samples and lines and units and spectra_names and wavelength):
                    raise Exception(f"Unsupported file: input file `{ENVI_RSR_DIR}/{self.sensor}.sli` does not follow the correct format!")
                hdr['samples'] = int(samples['samples']) # band_count
                hdr['lines']   = int(lines['lines']) # line_count
                hdr['data_type'], hdr['data_bytes'] = self._data_type(data_type['data_type'])
                hdr['units']   = units['units'] # units
                # hdr['factor']  = float(factor['factor']) # factor
                hdr['spectra_names'] = list(map(lambda x:x.strip(), spectra_names['spectra_names'].split(',')))
                hdr['wavelength'] = np.asarray(list(map(lambda x:float(x.strip()), wavelength['wavelength'].split(','))))

            # read sli
            self.band_count = hdr['lines']
            with open(f'{ENVI_RSR_DIR}/{self.sensor}.sli', 'rb') as f:
                f.seek(0, 0)
                idx = 0
                while True:
                    sig_channel = f.read(hdr['data_bytes'] * hdr['samples'])
                    if len(sig_channel) == 0:
                        break
                    else:
                        x = hdr['wavelength']
                        y = np.frombuffer(unhexlify(sig_channel.hex()), hdr['data_type'])
                        self.bands.append(Band(x, y, self.sensor, idx + 1, unit='um', desc=hdr['spectra_names'][idx]))
                        idx += 1
        # NWP source
        elif self._source == 'NWP':
            metadata = NWP_RSR_CONFIG[self.sensor]
            _band_prefix = metadata['band_prefix']
            skip_header = metadata['skip_header']
            self.band_count  = metadata['band_number']
            wave_units  = metadata['wave_units']
            for iband in range(self.band_count):
                band_name = _band_prefix % (iband + 1)
                data = np.genfromtxt(f"{NWP_RSR_DIR}/{self.sensor}/{band_name}.txt", skip_header=skip_header)
                if wave_units == 'wn':
                    data[:, 0] = np.array(list(map(convert_wave_unit, data[:, 0])))
                data = np.flipud(data)
                x = data[:, 0]
                y = data[:, 1]
                self.bands.append(Band(x, y, self.sensor, iband + 1))
        elif self._source == 'HDF5':
            filename = HDF5_RSR_DIR + f'rsr_{self.sensor}.h5'
            no_detectors_message = False
            import h5py
            with h5py.File(filename, 'r') as h5f:
                band_names = h5f.attrs['band_names'].tolist()
                self.band_count = len(band_names)
                description = h5f.attrs['description']
                if not isinstance(band_names[0], str):
                    band_names = [x.decode('utf-8') for x in band_names]

                for iband, bandname in enumerate(band_names):
                    try:
                        num_of_det = h5f[bandname].attrs['number_of_detectors']
                    except KeyError:
                        num_of_det = 1
                    # print(num_of_det)
                    for i in range(1, 2):
                        dname = 'det-{0:d}'.format(i)
                        try:
                            resp = h5f[bandname][dname]['response'][:]
                        except KeyError:
                            resp = h5f[bandname]['response'][:]
                        # The wavelength is given in micro meters!
                        try:
                            wvl = (h5f[bandname][dname]['wavelength'][:] *
                                   h5f[bandname][dname]['wavelength'].attrs['scale'])
                        except KeyError:
                            wvl = (h5f[bandname]['wavelength'][:] *
                                   h5f[bandname]['wavelength'].attrs['scale'])
                        try:
                            central_wvl = h5f[bandname][dname].attrs['central_wavelength']
                        except KeyError:
                            central_wvl = h5f[bandname].attrs['central_wavelength']
                    self.bands.append(Band(wvl * 1e6, resp, self.sensor, iband + 1, center=central_wvl, unit='um', desc=description))
        else:
            self.bands = []
            unit = None
            center = 0
            fwhm   = 0
            with open(USER_RSR_DIR + f'{self.sensor}.flt', "r") as f:
                inline = False # recording
                sig_band = []
                for line in f.readlines():
                    if not unit and 'Micrometer' in line:
                        unit = 'um'
                    elif not unit and 'Nanometer' in line:
                        unit = 'nm'
                    line = line.replace('CENTER', 'Center')
                    if line.find("Center") != -1:
                        info = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)
                        assert len(info) < 4
                        center = None if len(info) < 2 else float(info[-2])
                        fwhm = None if len(info) < 2 else float(info[-1])
                        if sig_band:
                            band_number = len(self.bands) + 1
                            data = np.asarray(sig_band)
                            self.bands.append(Band(data[:, 0], data[:, 1], sensor, band_number, center, fwhm, unit))
                        inline = True
                        sig_band = []
                        continue
                    elif not inline:
                        continue
                    else:
                        sig_band.append(list(map(lambda x:float(x), line.strip().split())))
                self.band_count = len(self.bands) + 1
                data = np.asarray(sig_band)
                self.bands.append(Band(data[:, 0], data[:, 1], sensor, band_number + 1, center, fwhm, unit))

    def integral(self, bandname):
        """Calculate the integral of the spectral response function for each detector."""
        intg = {}
        for det in self.rsr[bandname].keys():
            wvl = self.rsr[bandname][det]['wavelength']
            resp = self.rsr[bandname][det]['response']
            intg[det] = np.trapz(resp, wvl)
        return intg

    def convert(self):
        """Convert spectral response functions from wavelength to wavenumber."""
        if self._wavespace == WAVE_LENGTH:
            rsr, info = convert2wavenumber(self.rsr)
            for band in rsr.keys():
                for det in rsr[band].keys():
                    self.rsr[band][det][WAVE_NUMBER] = rsr[
                        band][det][WAVE_NUMBER]
                    self.rsr[band][det]['response'] = rsr[
                        band][det]['response']
                    self.unit = info['unit']
                    self.si_scale = info['si_scale']
            self._wavespace = WAVE_NUMBER
            for band in rsr.keys():
                for det in rsr[band].keys():
                    self.rsr[band][det]['central_wavenumber'] = \
                        get_central_wave(self.rsr[band][det][WAVE_NUMBER], self.rsr[band][det]['response'])
                    del self.rsr[band][det][WAVE_LENGTH]
        else:
            errmsg = "Conversion from {wn} to {wl} not supported yet".format(wn=WAVE_NUMBER, wl=WAVE_LENGTH)
            raise NotImplementedError(errmsg)

    def set_description(self, h5f):
        """Set the description."""
        self.description = h5f.attrs['description']
        self.description = convert2str(self.description)

    def set_band_names(self, h5f):
        """Set the band names."""
        self.band_names = h5f.attrs['band_names']
        self.band_names = [convert2str(x) for x in self.band_names]

    def set_sensor(self, h5f):
        """Set the sensor name."""
        if self.sensor:
            return

        try:
            self.sensor = h5f.attrs['sensor']
            self.sensor = convert2str(self.sensor)
        except KeyError:
            LOG.warning("No sensor name specified in HDF5 file")
            self.sensor = INSTRUMENTS.get(self.platform_name)

    def set_platform_name(self, h5f):
        """Set the platform name."""
        if self.platform_name:
            return

        try:
            self.platform_name = h5f.attrs['platform_name']
            self.platform_name = convert2str(self.platform_name)
        except KeyError:
            LOG.warning("No platform_name in HDF5 file")
            try:
                satname = h5f.attrs['platform']
                satname = convert2str(satname)
                sat_number = h5f.attrs['sat_number']
                self.platform_name = satname + '-' + str(sat_number)
            except KeyError:
                LOG.warning(
                    "Unable to determine platform name from HDF5 file content")
                self.platform_name = None

        self.platform_name = OSCAR_PLATFORM_NAMES.get(self.platform_name, self.platform_name)

    def get_number_of_detectors4bandname(self, h5f, bandname):
        """For a band name get the number of detectors, if any."""
        try:
            num_of_det = h5f[bandname].attrs['number_of_detectors']
        except KeyError:
            LOG.debug("No detectors found - assume only one...")
            num_of_det = 1

        return num_of_det

    def get_wv_bounds(self, unit="um"):
        """Get the wavelength bounds of the spectral response function."""
        max_wv, min_wv = None, None
        for iband in self.selected_bands:
            wv = self.bands[iband-1].get_wv(unit)
            if max_wv is None:
                max_wv = wv.max()
            if min_wv is None:
                min_wv = wv.min()
            max_wv = max(max_wv, wv.max())
            min_wv = min(min_wv, wv.min())
        return max_wv, min_wv

    def get_unit_symbol(self):
        return get_unit_symbol(self.unit)

    def set_band_responses_per_detector(self, h5f, bandname, detector_name):
        """Set the RSR responses for the band and detector."""
        self.rsr[bandname][detector_name] = {}
        try:
            resp = h5f[bandname][detector_name]['response'][:]
        except KeyError:
            resp = h5f[bandname]['response'][:]

        self.rsr[bandname][detector_name]['response'] = resp

    def set_band_wavelengths_per_detector(self, h5f, bandname, detector_name):
        """Set the RSR wavelengths for the band and detector."""
        try:
            wvl = (h5f[bandname][detector_name]['wavelength'][:] *
                   h5f[bandname][detector_name]['wavelength'].attrs['scale'])
        except KeyError:
            wvl = (h5f[bandname]['wavelength'][:] *
                   h5f[bandname]['wavelength'].attrs['scale'])

        # The wavelength is given in micro meters!
        self.rsr[bandname][detector_name]['wavelength'] = wvl * 1e6

    def set_band_central_wavelength_per_detector(self, h5f, bandname, detector_name):
        """Set the central wavelength for the band and detector."""
        try:
            central_wvl = h5f[bandname][detector_name].attrs['central_wavelength']
        except KeyError:
            central_wvl = h5f[bandname].attrs['central_wavelength']

        self.rsr[bandname][detector_name]['central_wavelength'] = central_wvl

    def get_relative_spectral_responses(self, h5f):
        """Read the rsr data and add to the object."""
        for bandname in self.band_names:
            self.rsr[bandname] = {}

            num_of_det = self.get_number_of_detectors4bandname(h5f, bandname)
            for i in range(1, num_of_det + 1):
                dname = 'det-{0:d}'.format(i)
                self.set_band_responses_per_detector(h5f, bandname, dname)
                self.set_band_wavelengths_per_detector(h5f, bandname, dname)
                self.set_band_central_wavelength_per_detector(h5f, bandname, dname)

    @property
    def selected_band_count(self):
        return len(self.selected_bands)
    @property
    def identifier(self):
        if self.selected_band_count == self.band_count:
            return self.sensor
        return self.sensor + '_B' + '_B'.join(list(map(str, self.selected_bands)))

    @property
    def fwhms(self):
        fwhms = []
        for iband in self.selected_bands:
            fwhms.append(self.bands[iband-1].fwhm)
        return np.asarray(fwhms)

    @property
    def centers(self):
        centers = []
        for iband in self.selected_bands:
            centers.append(self.bands[iband-1].center)
        return np.asarray(centers)

    def subset(self, selected_band):
        ''' Subset bands
        '''
        if isinstance(selected_band, int):
            selected_band = (selected_band, )
        for iband in selected_band:
            self._validate_value(iband)
        self.selected_bands = selected_band
        # self.generate_lut()

    def save(self, saved_path=None, selected_band=None, ):
        '''Save spectral response function to file
        '''
        if isinstance(selected_band, int):
            selected_band = (selected_band, )
        if selected_band is None:
            selected_band = self.selected_bands
        if saved_path:
            output_flt_file = f'{saved_path}/{self.identifier}.flt'
        else:
            output_flt_file = USER_RSR_DIR + f'{self.identifier}.flt'
        header = f"{self.bands[0].get_unit()} data for Sensor `{self.sensor}`, created at {datetime.now()}."
        with open(output_flt_file, "w") as dst:
            dst.write(f'{header}\n')
            for iband in selected_band:
                self._validate_value(iband)
                dst.write(str(self.bands[iband - 1]))
        return output_flt_file.replace(os.path.sep, '/')
        # np.savetxt(saved_file, self.array, fmt="%.6f", delimiter=" ")

    # def saveall(self, saved_file=None):
    # 	''' Save spectral response function to file
    # 	'''
    # 	if saved_file is None:
    # 		output_flt_file = USER_RSR_DIR + f'{self.identifier}.flt'
    # 	else:
    # 		output_flt_file = saved_file

    # 	header = f"{self.bands[0].get_unit()} data for Sensor `{self.sensor}`, created at {datetime.now()}."
    # 	with open(output_flt_file, "w") as dst:
    # 		dst.write(f'{header}\n')
    # 		for iband in self.selected_bands:
    # 			self._validate_value(iband)
    # 			dst.write(str(self.bands[iband - 1]))

    def bt2r(self, bt, surjection=False):
        ''' Convert bright temperature to radiance
        surjection: if True, return the surjection of bt to r
        '''
        if np.isnan(bt).any():
            return [np.nan] * self.selected_band_count
        r = []
        if isinstance(bt, abc.Iterable):
            if surjection:
                assert(len(bt) == self.selected_band_count)
                for i, iband in enumerate(self.selected_bands):
                    r.append(self.bands[iband-1].bt2r(bt[i]))
            else:
                # for i, iband in enumerate(self.selected_bands):
                #     pool = ThreadPool(12)
                #     result = pool.map(self.bands[iband - 1].bt2r, bt)
                #     pool.close()
                #     pool.join()
                #     r.append(result)
                # r.append(list(map(self.bands[iband-1].bt2r, bt)))
                for i, iband in enumerate(self.selected_bands):
                    ufunc = np.frompyfunc(self.bands[iband - 1].bt2r, 1, 1)
                    r.append(ufunc(bt).astype(np.float64))
                r = np.array(r).T
        else:
            for i, iband in enumerate(self.selected_bands):
                r.append(self.bands[iband-1].bt2r(bt))
        return np.array(r)

    def r2bt(self, radiance_array):
        """ Convert radiance to bright temperature
        """
        if np.isnan(radiance_array).any():
            return [np.nan] * self.selected_band_count
        BTs = []
        if len(np.array(radiance_array).shape) == 2:
            assert(np.array(radiance_array).shape[1] == self.selected_band_count)
            # for i, iband in enumerate(self.selected_bands):
            #     pool = ThreadPool(12)
            #     result = pool.map(self.bands[iband - 1].r2bt,
            #                       radiance_array[:, i])
            #     pool.close()
            #     pool.join()
            #     BTs.append(result)
            #     BTs.append(list(map(self.bands[iband-1].r2bt, radiance_array[:, i])))
            for i, iband in enumerate(self.selected_bands):
                ufunc = np.frompyfunc(self.bands[iband - 1].r2bt, 1, 1)
                BTs.append(ufunc(radiance_array[:, i]).astype(np.float64))
            BTs = np.array(BTs).T
        else:
            assert(len(radiance_array) == self.selected_band_count)
            for i, iband in enumerate(self.selected_bands):
                BTs.append(self.bands[iband-1].r2bt(radiance_array[i]))
        return np.array(BTs)

    def generate_lut(self, saved_file=None):
        ''' Generate Look-Up Table
        '''
        BTs = np.linspace(MIN_TMP, MAX_TMP, MAX_STEPS)
        lut = [BTs, ]
        for iband in self.selected_bands:
            lut.append(self.bands[iband-1].radiances)
        self.lut = np.vstack(lut).transpose()
        if saved_file:
            np.savetxt(saved_file, self.lut, fmt="%.8f", delimiter=" ")
        return self.lut

    def plot(self, selected_bands=None, show_span=True, loc='outside'):
        '''Plot spectral response curve
        '''
        if isinstance(selected_bands, int):
            self._validate_value(selected_bands)
            self.bands[selected_bands - 1].plot(self.cmap(selected_bands - 1), show_span=show_span, subplot=False)
        else:
            left_xlim = []
            right_xlim = []
            if not selected_bands:
                selected_bands = self.selected_bands
            for iband in selected_bands:
                self._validate_value(iband)
                self.bands[iband - 1].plot(self.cmap(iband - 1), show_span=show_span, subplot=True)
                l, r = self.bands[iband-1].xlim
                left_xlim.append(l)
                right_xlim.append(r)
            self._show(loc, (min(left_xlim), max(right_xlim)))

    def subplot(self, start=1, end=1, show_span=True, loc='outside'):
        '''Plot subset of spectral response curve
        '''
        if end < start:
            raise ValueError(f'band start index must be smaller or equal to end index')
        if start < 1:
            raise ValueError(f'band start number must > 0')
        if end > self.band_count:
            raise ValueError(f'band end number is out of range, sensor {self.sensor} has {self.band_count} bands.')
        left_xlim = []
        right_xlim = []
        for iband in range(start, end + 1):
            self.bands[iband-1].plot(self.cmap(iband), show_span=show_span, subplot=True)
            l, r = self.bands[iband-1].xlim
            left_xlim.append(l)
            right_xlim.append(r)
        self._show(loc, (min(left_xlim), max(right_xlim)))

    def plotall(self, show_span=True, loc='outside'):
        '''Plot all spectral response curves
        '''
        self.plot(range(1, self.band_count + 1), show_span=show_span, loc=loc)

    def interp(self, spectral_wv, spectral_res, unit='nm', plot_fig=False):
        ''' spectral to band interpolation
        '''
        band_interp = []
        if plot_fig:
            plt.plot(spectral_wv, spectral_res)
        for index in self.selected_bands:
            band = self.bands[index-1]
            if plot_fig:
                plt.plot(band.wv, interp1d(spectral_wv, spectral_res, fill_value="extrapolate")(band.wv))
            band_interp.append(band.interp(spectral_wv, spectral_res, unit=unit))
        if plot_fig:
            plt.show()
            plt.close()
        return np.asarray(band_interp)

    def effective_value(self, spectral_wv, spectral_res, unit='nm', plot_fig=False):
        ''' spectral to band interpolation
        '''
        band_effective_value = []
        if plot_fig:
            plt.plot(spectral_wv, spectral_res)
        for index in self.selected_bands:
            band = self.bands[index-1]
            if plot_fig:
                plt.plot(band.wv, interp1d(spectral_wv, spectral_res, fill_value="extrapolate")(band.wv))
            band_effective_value.append(band.effective_value(spectral_wv, spectral_res, unit=unit))
        if plot_fig:
            plt.show()
            plt.close()
        return np.asarray(band_effective_value)

    def effective_lambda(self):
        """ Effective wavelength
        Returns:
            float: effective wavelength in mircometers
        """
        lambdas = []
        for iband in self.selected_bands:
            lambdas.append(self.bands[iband-1].effective_lambda())
        return np.asarray(lambdas, dtype=np.float64)


    def b_lambda(self):
        b_lambdas = []
        for iband in self.selected_bands:
            b_lambdas.append(self.bands[iband-1].b_lambda())
        return np.asarray(b_lambdas, dtype=np.float64)

    def derivation_ab(self):
        params = []
        for iband in self.selected_bands:
            params.append(self.bands[iband-1].derivation_ab())
        return np.asarray(params, dtype=np.float64)

    def __getitem__(self, band_number):
        self._validate_value(band_number)
        return self.bands[band_number - 1]

    def __len__(self):
        return self.band_count

    def __iter__(self):
        self.current_index = -1
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.selected_band_count:
            raise StopIteration
        return self.bands[self.selected_bands[self.current_index] - 1]

    def __repr__(self):
        return f"Filter data for sensor {self.sensor}; created by simpir.\n"

    def _show(self, loc, xlim=[0, 1]):
        plt.xlabel(f'WaveLength({get_unit_symbol(self.unit)})')
        plt.ylabel('Spectral Response')
        if loc == 'outside':
            plt.legend(scatterpoints=1, labelspacing=0.4, columnspacing=2, markerscale=2, bbox_to_anchor=(1.01, 1.02), ncol=2)
        else:
            plt.legend(loc=loc)
        plt.ylim(0, 1.05)
        plt.xlim(*xlim)
        plt.title(f' "{self.sensor}" WL-SR Relationship')
        plt.show()
        plt.close()

    def _validate_value(self, band_number):
        if band_number == 0:
            raise ValueError(f'band number is out of range, band_number subscripts starting with one.')
        elif band_number > len(self.bands) or band_number < 1:
            raise ValueError(f'band number is out of range, sensor {self.sensor} has {self.band_count} bands.')
        return True

    def _data_type(self, code):
        """ Convert ENVI hdf data_type to Numpy array's dtype
        bool	布尔类型，true，false
        int_	默认的整数类型（类似于 C 语言中的 long，int32 或 int64）
        intc	与C的int类型一样，一般是 int32 或 int64
        intp	用于索引的整数类型（类似于 C 的 ssize_t，一般情况下仍然是 int32 或 int64）
        int8	字节（-128 to 127）
        int16	整数（-32768 to 32767）
        int32	整数（-2147483648 to 2147483647）
        int64	整数（-9223372036854775808 to 9223372036854775807）
        uint8	无符号整数（0 to 255）
        uint16	无符号整数（0 to 65535）
        uint32	无符号整数（0 to 4294967295）
        uint64	无符号整数（0 to 18446744073709551615）
        float_	float64 类型的简写
        float16	半精度浮点数，包括：1 个符号位，5 个指数位，10 个尾数位
        float32	单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位
        float64	双精度浮点数，包括：1 个符号位，11 个指数位，52 个尾数位
        complex_ complex128	类型的简写，即 128 位复数
        complex64	复数，表示双 32 位浮点数（实数部分和虚数部分）
        complex128	复数，表示双 64 位浮点数（实数部分和虚数部分）
        """
        if code == '1':
            return np.uint8, 1
        elif code == '2':
            return np.int16, 2
        elif code == '12':
            return np.uint16, 2
        elif code == '3':
            return np.int32, 4
        elif code == '13':
            return np.uint32, 4
        elif code == '4':
            return np.float32, 4
        elif code == '5':
            return np.float64, 8
        else:
            return np.float_, 8

RSR = RelativeSpectralResponse

if __name__ == '__main__':
    print(RSR_LIB)
    rsr = RelativeSpectralResponse('olci_Sentinel-3B')
    rsr.plot()