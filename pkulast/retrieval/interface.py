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
import time
import wrapt
import joblib
import math
import warnings
import logging
import tempfile
import rasterio
import functools
import concurrent
import numpy as np
import xarray as xr
from collections.abc import Iterable
from shapely import speedups

speedups.disable()

from pkulast.raster import applier
from pkulast.utils.io import check_filename_exist
from pkulast.utils.raster import save_array, get_tiles
from pkulast.retrieval.algorithm import SW, TES, LookupTable, TESv2, TESv3
from pkulast.exceptions import *
from pkulast.atmosphere.profile import NWPLibrary
from pkulast.rtm.model import run_modtran
from pkulast.surface.emissivity import UniRel, MultiRel
from pkulast.config import NUM_WORKERS, BLOCK_SIZE

warnings.filterwarnings('ignore')

LOG = logging.getLogger(__name__)


class Timer:
    timings = {}
    enabled = False

    def __init__(self, enabled=True):
        Timer.enabled = enabled

    def __str__(self):
        return "Timer"

    __repr__ = __str__

    @classmethod
    def timeit(cls,
               level=0,
               func_name=None,
               cls_name=None,
               prefix="[Method] "):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            if not cls.enabled:
                return func(*args, **kwargs)
            if instance is not None:
                instance_name = "{:>18s}".format(instance.__class__.__name__)
            else:
                instance_name = " " * 18 if cls_name is None else "{:>18s}".format(
                    cls_name)
            _prefix = "{:>26s}".format(prefix)
            try:
                _func_name = "{:>28}".format(
                    func.__name__ if func_name is None else func_name)
            except AttributeError:
                str_func = str(func)
                _at_idx = str_func.rfind("at")
                _dot_idx = str_func.rfind(".", None, _at_idx)
                _func_name = "{:>28}".format(str_func[_dot_idx + 1:_at_idx -
                                                      1])
            _name = instance_name + _prefix + _func_name
            _t = time.time()
            rs = func(*args, **kwargs)
            _t = time.time() - _t
            try:
                cls.timings[_name]["timing"] += _t
                cls.timings[_name]["call_time"] += 1
            except KeyError:
                cls.timings[_name] = {
                    "level": level,
                    "timing": _t,
                    "call_time": 1
                }
            return rs

        return wrapper

    @classmethod
    def show_log(cls, level=2):
        print()
        print("=" * 110 + "\n" + "Timer log\n" + "-" * 110)
        if cls.timings:
            for key in sorted(cls.timings.keys()):
                timing_info = cls.timings[key]
                if level >= timing_info["level"]:
                    print("{:<42s} :  {:12.7} s (Call Time: {:6d})".format(
                        key, timing_info["timing"], timing_info["call_time"]))
        print("-" * 110)

    @classmethod
    def disable(cls):
        cls.enabled = False


def block_operator(input, func, **kwargs):
    input = np.asarray(input)
    band, row, col = input.shape
    xds = xr.DataArray(input,
                       dims=['band', 'x', 'y'],
                       coords={
                           'band': range(band),
                           'x': range(row),
                           'y': range(col)
                       }).chunk({
                           'x': 128,
                           'y': 128
                       })
    res = xr.apply_ufunc(functools.partial(func, **kwargs),
                         xds,
                         input_core_dims=[["band"]],
                         output_dtypes=list,
                         dask='parallelized',
                         vectorize=True).compute()
    return np.asarray(res.values.tolist(), dtype=np.float32).transpose(
        (2, 0, 1))


class Executor:
    ''' caution: func, and kwargs must be serializable
	'''
    timer = Timer()

    def __init__(self, infile, outfile, **kwargs):
        self.outfile = outfile
        self.kwargs = kwargs
        self.isfile = False
        if isinstance(infile, str):
            if not check_filename_exist(infile):
                raise SimpirFileNotFoundException(f"no such a file: {infile}")
            else:
                self.isfile = True
                self.infile = infile
        else:
            filename = os.path.join(tempfile.gettempdir(),
                                    os.urandom(24).hex())
            save_array(infile, filename)
            self.infile = filename

    @timer.timeit(prefix="[run]")
    def run(self, func):
        with rasterio.Env():
            with rasterio.open(self.infile) as src:
                profile = src.profile
                if 'count' in self.kwargs:
                    count = self.kwargs['count']
                else:
                    count = profile['count']
                profile.update(compress='lzw',
                               blockxsize=BLOCK_SIZE,
                               blockysize=BLOCK_SIZE,
                               count=count,
                               tiled=True,
                               dtype=np.float64)
                with rasterio.open(self.outfile, 'w', **profile) as dst:
                    windows = [window for ij, window in dst.block_windows()]
                    # for window, result in zip(windows, joblib.Parallel(n_jobs=-1)(joblib.delayed(func)(src.read(window=window), **self.kwargs) for window in windows)):
                    #     dst.write(result, window=window)
                    data_gen = (src.read(window=window) for window in windows)
                    with concurrent.futures.ProcessPoolExecutor(
                            max_workers=NUM_WORKERS) as executor:
                        for window, result in zip(
                                windows,
                                executor.map(
                                    functools.partial(func, **self.kwargs),
                                    data_gen)):
                            dst.write(result, window=window)
                if not self.isfile:
                    os.remove(self.infile)


def atm_corr(infile,
             outfile,
             uward,
             trans,
             selected_bands=None,
             show_log=True,
             **kwargs):
    if selected_bands:
        assert uward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    else:
        selected_bands = np.arange(uward.shape[0])
    count = len(selected_bands)
    exec = Executor(infile,
                    outfile,
                    uward=uward,
                    trans=trans,
                    count=count,
                    selected_bands=selected_bands,
                    **kwargs)
    exec.run(_atm_corr)
    if show_log:
        exec.timer.show_log()


def calibration(infile,
                outfile,
                gains,
                offsets,
                selected_bands=None,
                show_log=True,
                **kwargs):
    if selected_bands:
        assert np.array(gains).shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    else:
        selected_bands = np.arange(gains.shape[0])
    count = len(selected_bands)
    exec = Executor(infile,
                    outfile,
                    gains=gains,
                    offsets=offsets,
                    count=count,
                    selected_bands=selected_bands,
                    **kwargs)
    exec.run(_calibration)
    if show_log:
        exec.timer.show_log()


def _calibration(input, gains, offsets, **kwargs):
    return input * gains[:, np.newaxis, np.newaxis] + offsets[:, np.newaxis,
                                                              np.newaxis]


def _atm_corr(input, uward, trans, **kwargs):
    return (input - uward[:, np.newaxis, np.newaxis]) / trans[:, np.newaxis,
                                                              np.newaxis]


class PixelExecutor(Executor):
    ''' caution: func, and kwargs must be serializable
	'''
    timer = Timer()

    @timer.timeit(prefix="[run]")
    def run(self, func):
        with rasterio.Env():
            with rasterio.open(self.infile) as src:
                profile = src.profile
                if 'count' in self.kwargs:
                    count = self.kwargs['count']
                else:
                    count = profile['count']
                profile.update(compress='lzw',
                               blockxsize=BLOCK_SIZE,
                               blockysize=BLOCK_SIZE,
                               count=count,
                               tiled=True,
                               dtype=np.float32)
                with rasterio.open(self.outfile, 'w', **profile) as dst:
                    windows = [window for ij, window in dst.block_windows()]
                    # for window, result in zip(windows, joblib.Parallel(n_jobs=-1)(joblib.delayed(block_operator)(src.read(window=window), func=func, **self.kwargs) for window in windows)):
                    # 	dst.write(result, window=window)
                    data_gen = (src.read(window=window) for window in windows)
                    with concurrent.futures.ProcessPoolExecutor(
                            max_workers=NUM_WORKERS) as executor:
                        for window, result in zip(
                                windows,
                                executor.map(
                                    functools.partial(block_operator,
                                                      func=func,
                                                      **self.kwargs),
                                    data_gen)):
                            dst.write(result, window=window)
                if not self.isfile:
                    os.remove(self.infile)


class RowExecutor(Executor):
    timer = Timer()

    @timer.timeit(prefix="[run]")
    def run(self, func):
        with rasterio.Env():
            with rasterio.open(self.infile) as src:
                profile = src.profile
                width = self.kwargs['width'] if 'width' else src.width
                profile.update(compress='lzw',
                               blockxsize=BLOCK_SIZE,
                               blockysize=BLOCK_SIZE,
                               width=width,
                               tiled=True,
                               dtype=np.float64)
                with rasterio.open(self.outfile, "w", **profile) as dst:
                    _windows = [
                        window for window, transform in get_tiles(
                            src, width=src.meta['width'], height=1)
                    ]
                    data_gen = (src.read(window=window) for window in _windows)
                    big_window = rasterio.windows.Window(
                        col_off=0,
                        row_off=0,
                        width=width,
                        height=src.meta['height'])
                    new_windows = (rasterio.windows.Window(
                        col_off=_window.col_off,
                        row_off=_window.row_off,
                        width=width,
                        height=1).intersection(big_window)
                                   for _window in _windows)
                    with concurrent.futures.ProcessPoolExecutor(
                            max_workers=NUM_WORKERS) as executor:
                        tasks = {
                            executor.submit(func, data, **self.kwargs): _window
                            for _window, data in zip(new_windows, data_gen)
                        }
                        for future in concurrent.futures.as_completed(tasks):
                            _window = tasks[future]
                            try:
                                result = future.result()
                                dst.write(result, window=_window)
                            except Exception as exc:
                                print('%r generated an exception: %s' %
                                      (_window, exc))
        end = time.time()
        if not self.isfile:
            os.remove(self.infile)


class ColExecutor(Executor):
    timer = Timer()

    @timer.timeit(prefix="[run]")
    def run(self, func):
        with rasterio.Env():
            with rasterio.open(self.infile) as src:
                profile = src.profile
                height = self.kwargs[
                    'height'] if 'height' in self.kwargs else src.height
                profile.update(compress='lzw',
                               blockxsize=BLOCK_SIZE,
                               blockysize=BLOCK_SIZE,
                               height=height,
                               tiled=True,
                               dtype=np.float32)
                with rasterio.open(self.outfile, "w", **profile) as dst:
                    _windows = [
                        window for window, transform in get_tiles(
                            src, width=1, height=src.meta['height'])
                    ]
                    data_gen = (src.read(window=window) for window in _windows)
                    big_window = rasterio.windows.Window(
                        col_off=0,
                        row_off=0,
                        width=src.meta['width'],
                        height=height)
                    new_windows = (rasterio.windows.Window(
                        col_off=_window.col_off,
                        row_off=_window.row_off,
                        width=1,
                        height=height).intersection(big_window)
                                   for _window in _windows)
                    with concurrent.futures.ProcessPoolExecutor(
                            max_workers=NUM_WORKERS) as executor:
                        tasks = {
                            executor.submit(func, data, **self.kwargs): _window
                            for _window, data in zip(new_windows, data_gen)
                        }
                        for future in concurrent.futures.as_completed(tasks):
                            _window = tasks[future]
                            try:
                                result = future.result()
                                dst.write(result, window=_window)
                            except Exception as exc:
                                print('%r generated an exception: %s' %
                                      (_window, exc))
        # clean up
        if not self.isfile:
            os.remove(self.infile)


def pixel_r2bt(infile, outfile, f, show_log=True):
    """ radiance to bt
	"""
    exec = PixelExecutor(infile, outfile)
    exec.run(f.r2bt)
    if show_log:
        exec.timer.show_log()


def pixel_bt2r(infile, outfile, f, show_log=True):
    """ bt to radiance
	"""
    exec = PixelExecutor(infile, outfile)
    exec.run(f.bt2r)
    if show_log:
        exec.timer.show_log()


def aster_pixel_atm_corr(infile,
                         outfile,
                         band_count,
                         selected_bands=None,
                         show_log=True,
                         **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(band_count))
    else:
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    count = len(selected_bands)
    exec = PixelExecutor(infile,
                         outfile,
                         count=count,
                         band_count=band_count,
                         selected_bands=selected_bands,
                         **kwargs)
    exec.run(_aster_pixel_atm_corr)
    if show_log:
        exec.timer.show_log()


def _aster_pixel_atm_corr(radiance, band_count, selected_bands, **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(band_count))
    else:
        selected_bands = np.asarray(selected_bands, dtype=int)
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands), 0)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = (radiance[selected_bands + 2 * band_count] -
                   radiance[selected_bands]) / radiance[selected_bands +
                                                        band_count]
        return ret


def aster_pixel_tes(infile,
                    outfile,
                    band_count,
                    lut,
                    emis_func,
                    selected_bands=None,
                    show_log=True,
                    **kwargs):
    """ TES algorithm
	"""
    if selected_bands:
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    else:
        selected_bands = np.arange(band_count)
    if lut.shape[1] != len(selected_bands) + 1:
        lut = lut[:, np.concatenate([[0], selected_bands + 1])]
    count = band_count + 1
    exec = PixelExecutor(infile,
                         outfile,
                         count=count,
                         band_count=band_count,
                         lut=lut,
                         emis_func=emis_func,
                         selected_bands=selected_bands,
                         **kwargs)
    print("Executing TES Version: {}".format(kwargs['tes_version']))
    if kwargs['tes_version'] == 'v3':
        exec.run(_aster_pixel_tes_v3)
    elif kwargs['tes_version'] == 'v2':
        exec.run(_aster_pixel_tes_v2)
    else:
        exec.run(_aster_pixel_tes)
    if show_log:
        exec.timer.show_log()


def _aster_pixel_tes(radiance, band_count, lut, emis_func, selected_bands,
                     **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(band_count))
    else:
        selected_bands = np.asarray(selected_bands, dtype=int)
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands) + 1, 0)
    tes = TES(lut, emis_func)
    return tes.compute(radiance[selected_bands + band_count],
                       radiance[selected_bands])


def _aster_pixel_tes_v2(radiance, band_count, lut, emis_func, selected_bands,
                        **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(band_count))
    else:
        selected_bands = np.asarray(selected_bands, dtype=int)
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands) + 1, 0)
    tes = TESv2(lut, emis_func)
    return tes.compute(radiance[selected_bands + band_count],
                       radiance[selected_bands])


def _aster_pixel_tes_v3(radiance, band_count, lut, emis_func, selected_bands,
                        **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(band_count))
    else:
        selected_bands = np.asarray(selected_bands, dtype=int)
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands) + 1, 0)
    tes = TESv3(lut, emis_func)
    return tes.compute(radiance[selected_bands + band_count],
                       radiance[selected_bands])


def pixel_tes(infile,
              outfile,
              dward,
              lut,
              emis_func,
              selected_bands=None,
              show_log=True,
              **kwargs):
    """ TES algorithm
	"""
    if selected_bands:
        assert dward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    else:
        selected_bands = np.arange(dward.shape[0])
    if lut.shape[1] != len(selected_bands) + 1:
        lut = lut[:, np.concatenate([[0], selected_bands + 1])]
    count = dward.shape[0] + 1
    exec = PixelExecutor(infile,
                         outfile,
                         count=count,
                         dward=dward,
                         lut=lut,
                         emis_func=emis_func,
                         selected_bands=selected_bands,
                         **kwargs)
    print("Executing TES Version: {}".format(kwargs['tes_version']))
    if kwargs['tes_version'] == 'v3':
        exec.run(_pixel_tes_v3)
    elif kwargs['tes_version'] == 'v2':
        exec.run(_pixel_tes_v2)
    else:
        exec.run(_pixel_tes)
    if show_log:
        exec.timer.show_log()


def pixel_sw(rsr,
             algorithm,
             infile,
             outfile,
             sw_params=None,
             sw_coeffs=None,
             selected_bands=None,
             show_log=True,
             **kwargs):
    """ SW algorithm
	"""
    sw = SW(algorithm=algorithm)
    if sw_coeffs is not None:
        sw.set_coeffs(sw_coeffs)
    else:
        if sw_params is not None:
            sw.set_params(rsr, **sw_params)
            sw.get_models()
        else:
            raise ValueError("No SW parameters provided")
    with rasterio.Env():
        with rasterio.open(infile) as src:
            profile = src.profile
            if selected_bands:
                selected_bands = np.asarray(selected_bands, dtype=int) - 1
            else:
                selected_bands = np.arange(profile.count)
            profile.update(compress='lzw',
                            blockxsize=BLOCK_SIZE,
                            blockysize=BLOCK_SIZE,
                            tiled=True,
                            dtype=np.float64)
            with rasterio.open(outfile, 'w', **profile) as dst:
                bt = src.read()[selected_bands]
                result = sw.compute(bt, **kwargs)
                dst.write(result)
    print("Executing SW Algorithm: {}".format(kwargs['sw_algorithm']))
    if show_log:
        exec.timer.show_log()


def pixel_radiometric_correction(infile,
                                 outfile,
                                 gains,
                                 offsets,
                                 selected_bands=None,
                                 show_log=True,
                                 **kwargs):
    exec = PixelExecutor(infile,
                         outfile,
                         gains=gains,
                         offsets=offsets,
                         selected_bands=None,
                         **kwargs)
    exec.run(_pixel_calibration)
    if show_log:
        exec.timer.show_log()


def pixel_atm_corr(infile,
                   outfile,
                   uward,
                   trans,
                   selected_bands=None,
                   show_log=True,
                   **kwargs):
    if selected_bands:
        assert uward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    else:
        selected_bands = np.arange(uward.shape[0])
    count = len(selected_bands)
    exec = PixelExecutor(infile,
                         outfile,
                         count=count,
                         uward=uward,
                         trans=trans,
                         selected_bands=selected_bands,
                         **kwargs)
    exec.run(_pixel_atm_corr)
    if show_log:
        exec.timer.show_log()


def pixel_simulation(f, profiles, emissivity, bt):
    pass


def pixel_validation():
    pass


def pixel_retrieval(infile,
                    outfile,
                    f,
                    lat,
                    lon,
                    acq_time,
                    is_satellite,
                    flight_height,
                    ground_height,
                    save_path,
                    selected_bands=None,
                    gains=None,
                    offsets=None,
                    calibrated=False,
                    atmcorrected=False,
                    show_fig=False,
                    emis_func=None,
                    tes_version='v1'):
    """ Retrieval
	"""
    start = time.time()
    # output configuration
    basename, extension = os.path.splitext(os.path.basename(infile))
    cbtcorr_file = os.path.join(save_path,
                                basename + "_calibrated_" + extension)
    atmcorr_file = os.path.join(save_path,
                                basename + "_atmremoved_" + extension)
    # 1. sensor filter
    if selected_bands is not None:
        f.subset(selected_bands)
    lut = f.generate_lut()
    if show_fig:
        f.plot()

    basename, extension = os.path.splitext(os.path.basename(outfile))
    outfile = os.path.join(save_path,
                           basename + '_' + f.identifier + extension)
    print(outfile)

    # 2. radiometric correction
    if not calibrated:
        assert len(gains) == f.band_count and len(offsets) == f.band_count
        pixel_radiometric_correction(infile,
                                     cbtcorr_file,
                                     gains=gains,
                                     offsets=offsets)
        infile = cbtcorr_file

    # 3. MMD fitting
    if emis_func is None:
        mni = UniRel(f)
        mni.fit()
        if show_fig:
            mni.plot()
        emis_func = mni.emis_func
    # 4. atmospheric correction and/or retrieval
    nwp = NWPLibrary()
    profile = nwp.extract(acq_time, lat, lon)
    if show_fig:
        profile.plot()
        # run_modtran(profiles, flt, is_satellite, flight_altitude=None, ground_altitude=None, mult= False, include_solar=False, vza=0, out_file=None)
    atm = run_modtran([
        profile,
    ], f, is_satellite, flight_height, ground_height)
    print(atm)
    uward = atm['uward'][0]
    dward = atm['dward'][0]
    trans = atm['trans'][0]
    if not atmcorrected:
        pixel_atm_corr(infile,
                       atmcorr_file,
                       uward=uward,
                       trans=trans,
                       selected_bands=selected_bands)
        infile = atmcorr_file
        pixel_tes(infile,
                  outfile,
                  dward=dward,
                  lut=lut,
                  emis_func=emis_func,
                  src_nodata=0,
                  tes_version=tes_version)
    else:
        pixel_tes(infile,
                  outfile,
                  dward=dward,
                  lut=lut,
                  selected_bands=selected_bands,
                  emis_func=emis_func,
                  src_nodata=0,
                  tes_version=tes_version)


class RIOSExecutor:
    timer = Timer()

    def __init__(self, infiles, outfiles, **kwargs):
        self.infiles = applier.FilenameAssociations()
        self.infiles.image = infiles
        self.outfiles = applier.FilenameAssociations()
        self.outfiles.outimage = outfiles
        self.otherargs = applier.OtherInputs()
        self.controls = applier.ApplierControls()
        # self.controls.setNumThreads(4)
        # self.controls.setJobManagerType('subproc')
        setattr(self.otherargs, 'kwargs', kwargs)
        for k, v in kwargs.items():
            setattr(self.otherargs, k, v)
        self._operator = None

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, op):
        ''' specific this function to customize operator
		'''
        self._operator = op

    @timer.timeit(prefix="[run]")
    def run(self):
        applier.apply(self._operator,
                      self.infiles,
                      self.outfiles,
                      self.otherargs,
                      controls=self.controls)


def RIOSCalibration(infile, outfile, gains, offsets, show_log=False):
    exec = RIOSExecutor(infile, outfile, gains=gains, offsets=offsets)
    exec.operator = _rios_calibration
    exec.run()
    if show_log:
        exec.timer.show_log()


def RIOSAtmCorrIR(infile, outfile, uward, trans, show_log=False):
    exec = RIOSExecutor(infile, outfile, uward=uward, trans=trans)
    exec.operator = _rios_atm_corr_ir
    exec.run()
    if show_log:
        exec.timer.show_log()


def RIOSRadiance2BT(infile, outfile, rsr, show_log=False):
    exec = RIOSExecutor(infile, outfile, rsr=rsr)
    exec.operator = _rios_radiance2bt
    exec.run()
    if show_log:
        exec.timer.show_log()


def RIOSBT2Radiance(infile, outfile, rsr, show_log=False):
    exec = RIOSExecutor(infile, outfile, rsr=rsr)
    exec.operator = _rios_bt2radiance
    exec.run()
    if show_log:
        exec.timer.show_log()


def _rios_calibration(info, inputs, outputs, otherargs):
    gains = otherargs.gains
    offsets = otherargs.offsets
    if isinstance(gains, Iterable):
        outputs.outimage = np.array(
            gains)[:, np.newaxis, np.newaxis] * inputs.image + np.array(
                offsets)[:, np.newaxis, np.newaxis]
    else:
        outputs.outimage = gains * inputs.image + offsets


def _rios_atm_corr_ir(info, inputs, outputs, otherargs):
    uward = otherargs.uward
    trans = otherargs.trans
    if isinstance(trans, Iterable):
        outputs.outimage = (inputs.image - uward[:, np.newaxis, np.newaxis]
                            ) / trans[:, np.newaxis, np.newaxis]
    else:
        outputs.outimage = (inputs.image - uward) / trans


def _rios_radiance2bt(info, inputs, outputs, otherargs):
    rsr = otherargs.rsr
    band, row, col = inputs.image.shape
    xds = xr.DataArray(inputs.image,
                       dims=['band', 'x', 'y'],
                       coords={
                           'band': range(band),
                           'x': range(row),
                           'y': range(col)
                       }).chunk({
                           'x': 64,
                           'y': 64
                       })
    res = xr.apply_ufunc(rsr.r2bt,
                         xds,
                         input_core_dims=[["band"]],
                         output_dtypes=list,
                         dask='parallelized',
                         vectorize=True).compute()
    outputs.outimage = np.asarray(res.values.tolist(),
                                  dtype=inputs.image.dtype).transpose(
                                      (2, 0, 1))


def _rios_bt2radiance(info, inputs, outputs, otherargs):
    ''' band * row * col dimensions array
	'''
    rsr = otherargs.rsr
    band, row, col = inputs.image.shape
    xds = xr.DataArray(inputs.image,
                       dims=['band', 'x', 'y'],
                       coords={
                           'band': range(band),
                           'x': range(row),
                           'y': range(col)
                       }).chunk({
                           'x': 64,
                           'y': 64
                       })
    res = xr.apply_ufunc(rsr.bt2r,
                         xds,
                         input_core_dims=[["band"]],
                         output_dtypes=list,
                         dask='parallelized',
                         vectorize=True).compute()
    outputs.outimage = np.asarray(res.values.tolist(),
                                  dtype=inputs.image.dtype).transpose(
                                      (2, 0, 1))


class RIOSPixelExecutor:
    timer = Timer()

    def __init__(self, infiles, outfiles, pixel_func, **kwargs):
        self.infiles = applier.FilenameAssociations()
        self.infiles.image = infiles
        self.outfiles = applier.FilenameAssociations()
        self.outfiles.outimage = outfiles
        self.otherargs = applier.OtherInputs()
        setattr(self.otherargs, 'kwargs', kwargs)
        setattr(self.otherargs, 'pixel_func', pixel_func)
        for k, v in kwargs.items():
            setattr(self.otherargs, k, v)

    @timer.timeit(prefix="[run]")
    def run(self):
        applier.apply(_rios_pixel_operator, self.infiles, self.outfiles,
                      self.otherargs)


def _rios_pixel_operator(info, inputs, outputs, otherargs):
    array = np.asarray(inputs.image)
    kwargs = otherargs.kwargs
    pixel_func = otherargs.pixel_func
    band, row, col = array.shape
    xds = xr.DataArray(array,
                       dims=['band', 'x', 'y'],
                       coords={
                           'band': range(band),
                           'x': range(row),
                           'y': range(col)
                       }).chunk({
                           'x': 128,
                           'y': 128
                       })
    res = xr.apply_ufunc(functools.partial(pixel_func, **kwargs),
                         xds,
                         input_core_dims=[["band"]],
                         output_dtypes=list,
                         dask='parallelized',
                         vectorize=True).compute()
    outputs.outimage = np.asarray(res.values.tolist(),
                                  dtype=np.float32).transpose((2, 0, 1))


def rios_pixel_tes(infile,
                   outfile,
                   dward,
                   lut,
                   emis_func,
                   selected_bands=None,
                   show_log=False,
                   **kwargs):
    pixel_exec = RIOSPixelExecutor(infile,
                                   outfile,
                                   _pixel_tes,
                                   dward=dward,
                                   lut=lut,
                                   selected_bands=selected_bands,
                                   emis_func=emis_func,
                                   **kwargs)
    pixel_exec.run()
    if show_log:
        pixel_exec.timer.show_log()


def rios_pixel_atm_corr(infile,
                        outfile,
                        uward,
                        trans,
                        selected_bands=None,
                        show_log=False,
                        **kwargs):
    pixel_exec = RIOSPixelExecutor(infile,
                                   outfile,
                                   _pixel_atm_corr,
                                   uward=uward,
                                   trans=trans,
                                   selected_bands=selected_bands,
                                   **kwargs)
    pixel_exec.run()
    if show_log:
        pixel_exec.timer.show_log()


def rios_pixel_calibration(infile,
                           outfile,
                           gains,
                           offsets,
                           selected_bands=None,
                           show_log=False,
                           **kwargs):
    pixel_exec = RIOSPixelExecutor(infile,
                                   outfile,
                                   _pixel_calibration,
                                   gains=gains,
                                   offsets=offsets,
                                   selected_bands=selected_bands,
                                   **kwargs)
    pixel_exec.run()
    if show_log:
        pixel_exec.timer.show_log()


def _pixel_tes(radiance, dward, lut, emis_func, selected_bands, **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(radiance.shape[0]))
    else:
        assert dward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int)
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands) + 1, 0)
    tes = TES(lut, emis_func)
    return tes.compute(radiance[selected_bands], dward)


def _pixel_tes_v2(radiance, dward, lut, emis_func, selected_bands, **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(radiance.shape[0]))
    else:
        assert dward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int)

    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands) + 1, 0)
    tes = TESv2(lut, emis_func)
    return tes.compute(radiance[selected_bands], dward)


def _pixel_tes_v3(radiance, dward, lut, emis_func, selected_bands, **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(radiance.shape[0]))
    else:
        assert dward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int)

    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands) + 1, 0)
    tes = TESv3(lut, emis_func)
    return tes.compute(radiance[selected_bands], dward)


def _pixel_atm_corr(radiance, uward, trans, selected_bands, **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(radiance.shape[0]))
    else:
        assert uward.shape[0] == len(selected_bands)
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        ret = np.full(len(selected_bands), 0)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = (radiance[selected_bands] - uward) / trans
        ret[ret < 0] = 0
    return ret


def _pixel_calibration(radiance, gains, offsets, selected_bands, **kwargs):
    if selected_bands is None:
        selected_bands = np.asarray(range(radiance.shape[0]))
    else:
        assert len(selected_bands) == np.asarray(gains).shape[0]
        selected_bands = np.asarray(selected_bands, dtype=int) - 1
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        ret = np.full(len(selected_bands), 0)
    else:
        ret = gains * radiance[selected_bands] + offsets
        if ('dst_nodata' in kwargs.keys()
                and np.any(ret < kwargs.get('dst_nodata'))):
            ret[ret < kwargs.get('dst_nodata')] = 0  # default nodata value
    return ret


def _pixel_emiss(radiance, bt, dward, **kwargs):
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(radiance) - 1, 0)
    return (radiance - dward) / (bt - dward)


def _pixel_calc_emis(radiance, uward, dward, trans, selected_bands, lut,
                     **kwargs):
    if np.isnan(radiance).any() or ('src_nodata' in kwargs.keys() and np.any(
            radiance <= kwargs.get('src_nodata'))):
        return np.full(len(selected_bands), 0)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            lst = radiance[0]
            mylut = LookupTable(lut)
            landrad = (radiance[selected_bands] * 0.01 - uward) / trans
            emis = (landrad - dward) / (mylut.bt2r(lst) - dward)
        emis[emis < kwargs.get('dst_nodata')] = 0
    return emis


###################################################################
# Split Window Implementation
###################################################################
def compute_cwv(
    IR_band1,
    IR_band2,
    cwv_coefs,
    window_size,
):
    """ Calculate the column water vapor (CWV) from the two IR bands.
    """
    assert (window_size % 2) == 1, "Window size should be a odd number"
    trans_mat_output = np.zeros(IR_band1.shape)  #Transmitance ratio
    padding_pixels = math.floor(window_size / 2)
    IR_band1_pad = np.pad(IR_band1, padding_pixels, 'reflect')
    IR_band2_pad = np.pad(IR_band2, padding_pixels, 'reflect')
    for i in range(0, IR_band1_pad.shape[0] - window_size + 1):
        for j in range(0, IR_band2_pad.shape[1] - window_size + 1):
            window_IR_band1 = IR_band1_pad[i:i + window_size,
                                           j:j + window_size]
            window_IR_band2 = IR_band2_pad[i:i + window_size,
                                           j:j + window_size]
            numerator = np.sum(
                np.multiply((window_IR_band1 - (np.median(window_IR_band1))),
                            (window_IR_band2 - (np.median(window_IR_band2)))))
            denominator = np.sum(
                (window_IR_band1 - np.median(window_IR_band1))**2)
            trans_mat_output[i, j] = numerator / denominator

    # C0 = 9.087
    # C1 = 0.653
    # C2 = -9.674
    C0, C1, C2 = cwv_coefs
    cwv_matrix = C0 + (C1 * trans_mat_output) + (C2 * (trans_mat_output**2))
    return cwv_matrix
