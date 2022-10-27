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
''' 
Empirical relationship for TIR spectrum of ground target.
	single variate regression:
		MMD: maximun-minimum difference
		MMR: maximun-minimum ratio
		VAR: variance
	multivariate regression
		MLR: multiple linear regression
		SVR: support vector regression
		GBT: gradient boost tree
'''
import os
import math
import glob
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from scipy import optimize
from collections import defaultdict

from pkulast.config import *
from pkulast.utils.spectrum import fractional_vegetation_cover, cavity_effect, rescale_band
from pkulast.utils.collections import get_cmap
from pkulast.utils.stats import get_rmse, get_bias, get_rsquare
from pkulast.utils.spectrum import RATIO


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams["font.size"] = 16
LOG = logging.getLogger(__name__)

def select_emissivity_library(rsr, emis_lib=DEFAULT_EMIS_LIB):
    ''' selected emissivity file for sensor's filter
	'''
    # UCSB .mht
    emis_files = []
    count = 0
    for ucsb_file in glob.glob(emis_lib + '/*.prn'):
        count += 1
        with open(ucsb_file, 'r') as fn:
            string = fn.readlines()
        offset = 0 if '#MEMO' in string else 1
        wv, _, emis = np.genfromtxt(ucsb_file,
               skip_header=PRN_HEADER - offset,
               skip_footer=PRN_FOOTER,
               encoding='utf-8').transpose()
        try:
            if np.min(rsr.interp(wv, emis, 'um')) >= 1:
                pass
            else:
                # # LOG.info(ucsb_file)
                emis_files.append(ucsb_file)
        except Exception as ex:
            pass
            # # LOG.info(ex)
    # ASTER(JPL JHU USGS) .txt
    for aster_file in glob.glob(emis_lib + '/*.spectrum.txt'):
        count += 1
        wv, ref = np.genfromtxt(aster_file,
              skip_header=ASTER_HEADER,
              skip_footer=ASTER_FOOTER).transpose()
        ref[ref < 0] = 0
        try:
            if np.min(rsr.interp(wv, 1 - ref / 100, 'um')) >= 1:
                # LOG.warning(aster_file + ' has emissivity value > 1')
                pass
            else:
                # # LOG.info(aster_file)
                emis_files.append(aster_file)
        except Exception as ex:
            pass
            # # LOG.info(ex)
    # LOG.info(f'{len(emis_files)} out of {count} samples were selected')
    return emis_files

def get_band_emissivity(rsr, emis_lib=DEFAULT_EMIS_LIB, save_path=None):#'./'
    ''' calculate band emissivity for sensor's filter
	'''
    # UCSB .mht
    band_emissvity = []
    emis_files = []
    count = 0
    for ucsb_file in glob.glob(emis_lib + '/*.prn'):
        count += 1
        with open(ucsb_file, 'r') as fn:
            string = fn.readlines()
        offset = 0 if '#MEMO' in string else 1
        wv, _, emis = np.genfromtxt(ucsb_file,
               skip_header=PRN_HEADER - offset,
               skip_footer=PRN_FOOTER,
               encoding='utf-8').transpose()
        try:
            if np.min(rsr.interp(wv, emis, 'um')) >= 1:
                pass
            else:
                # # LOG.info(ucsb_file)
                band_emissvity.append(rsr.interp(wv, emis, 'um'))
                emis_files.append(ucsb_file)
        except Exception as ex:
            pass
            # # LOG.info(ex)
    # ASTER(JPL JHU USGS) .txt
    for aster_file in glob.glob(emis_lib + '/*.spectrum.txt'):
        count += 1
        # print(aster_file)
        try:
            wv, ref = np.genfromtxt(aster_file,
                skip_header=ASTER_HEADER,
                skip_footer=ASTER_FOOTER).transpose()
            ref[ref < 0] = 0
        except Exception as ex:
            print(aster_file)
            continue
        try:
            if np.min(rsr.interp(wv, 1 - ref / 100, 'um')) >= 1:
                # LOG.warning(aster_file + ' has emissivity value > 1')
                pass
            else:
                # # LOG.info(aster_file)
                band_emissvity.append(rsr.interp(wv, 1 - ref / 100, 'um'))
                emis_files.append(aster_file)
        except Exception as ex:
            pass
            # # LOG.info(ex)
    if save_path:
        np.savetxt(f'{save_path}/{rsr.sensor}_band_emissvity.txt',
            band_emissvity,
            fmt='%10.18e')
    # LOG.info(f'{len(band_emissvity)} out of {count} samples were selected')
    return np.asarray(band_emissvity)

def get_spec_emissivity(rsr, emis_lib=DEFAULT_EMIS_LIB):
    ''' get spectral emissivity for sensor's filter
	'''
    # UCSB .mht
    spec_emissvity = []
    emis_files = []
    count = 0
    for ucsb_file in glob.glob(emis_lib + '/*.prn'):
        count += 1
        with open(ucsb_file, 'r') as fn:
            string = fn.readlines()
        offset = 0 if '#MEMO' in string else 1
        wv, _, emis = np.genfromtxt(ucsb_file,
               skip_header=PRN_HEADER - offset,
               skip_footer=PRN_FOOTER,
               encoding='utf-8').transpose()
        try:
            if np.min(rsr.interp(wv, emis, 'um')) >= 1:
                pass
            else:
                # # LOG.info(ucsb_file)
                # print(os.basename(ucsb_file))
                spec_emissvity.append((wv, emis))
                emis_files.append(ucsb_file)
        except Exception as ex:
            pass
            # # LOG.info(ex)
    # ASTER(JPL JHU USGS) .txt
    for aster_file in glob.glob(emis_lib + '/*.spectrum.txt'):
        count += 1
        wv, ref = np.genfromtxt(aster_file,
              skip_header=ASTER_HEADER,
              skip_footer=ASTER_FOOTER).transpose()
        ref[ref < 0] = 0
        try:
            if np.min(rsr.interp(wv, 1 - ref / 100, 'um')) >= 1:
                pass
                # LOG.warning(aster_file + ' has emissivity value > 1')
            else:
                # print(os.path.basename(aster_file))
                # # LOG.info(aster_file)
                spec_emissvity.append((wv, 1 - ref / 100))
                emis_files.append(aster_file)
        except Exception as ex:
            LOG.error(ex)
            pass
    # LOG.info(f'{len(emis_files)} out of {count} samples were selected')
    return spec_emissvity

def convolve_emissivity(rsr1, rsr2, band_values):
    rsr1_emis = get_band_emissivity(rsr1)
    rsr2_emis = get_band_emissivity(rsr2)
    x, residuals, rank, s = np.linalg.lstsq(rsr1_emis, rsr2_emis, rcond=None)
    return np.matmul(band_values, x), residuals, rank, s

def interp_emissivity(rsr, spec_file, unit='um', plot_fig=False):
    sample = np.genfromtxt(spec_file)
    return rsr.interp(sample[:, 0], sample[:, 1], unit=unit, plot_fig=plot_fig)


class Emissivity:
    def __init__(self):
        """Parent class for all emissivity methods. Contains general methods and attributes"""
        self.ndvi_min = -1
        self.ndvi_max = 1
        self.baresoil_ndvi_max = 0.2
        self.vegatation_ndvi_min = 0.5

    def __call__(self, **kwargs) -> np.ndarray:
        """Computes the emissivity
        kwargs:
        **ndvi (np.ndarray): NDVI image
        **red_band (np.ndarray): Band 4 or Red band image.
        **mask (np.ndarray[bool]): Mask image. Output will have NaN value where mask is True.
        Returns:
            Tuple(np.ndarray, np.ndarray): Emissivity for bands 10 and 11 respectively
        """
        if "ndvi" not in kwargs:
            raise ValueError("NDVI image is not provided")

        if "red_band" not in kwargs:
            raise ValueError("Band 4 (red band) image is not provided")

        self.ndvi = kwargs["ndvi"]
        self.red_band = kwargs["red_band"]

        if (self.ndvi is not None and self.red_band is not None
                and self.ndvi.shape != self.red_band.shape):
            raise ValueError(
                "Input images (NDVI and Red band) must be of equal dimension")

        emm_10, emm_11 = self._compute_emissivity()
        mask = emm_10 == 0
        emm_10[mask] = np.nan
        if emm_11 is not None:
            emm_11[mask] = np.nan
        return emm_10, emm_11

    def _compute_emissivity(self):
        raise NotImplementedError(
            "No concrete implementation of emissivity method yet")

    def _get_land_surface_mask(self):
        mask_baresoil = (self.ndvi >= self.ndvi_min) & (self.ndvi <
                                                        self.baresoil_ndvi_max)
        mask_vegetation = (self.ndvi > self.vegatation_ndvi_min) & (
            self.ndvi <= self.ndvi_max)
        mask_mixed = (self.ndvi >= self.baresoil_ndvi_max) & (
            self.ndvi <= self.vegatation_ndvi_min)
        return {
            "baresoil": mask_baresoil,
            "vegetation": mask_vegetation,
            "mixed": mask_mixed,
        }

    def _get_landcover_mask_indices(self):
        """Returns indices corresponding to the different landcover classes of of interest namely:
        vegetation, baresoil and mixed"
        """
        masks = self._get_land_surface_mask()
        baresoil = np.where(masks["baresoil"])
        vegetation = np.where(masks["vegetation"])
        mixed = np.where(masks["mixed"])
        return {"baresoil": baresoil, "vegetation": vegetation, "mixed": mixed}

    def _compute_fvc(self):
        # Returns the fractional vegegation cover from the NDVI image.
        return fractional_vegetation_cover(self.ndvi)


class ComputeMonoWindowEmissivity(Emissivity):

    emissivity_soil_10 = 0.97
    emissivity_veg_10 = 0.99
    emissivity_soil_11 = None
    emissivity_veg_11 = None

    def _compute_emissivity(self) -> np.ndarray:
        emm = np.empty_like(self.ndvi)
        landcover_mask_indices = self._get_landcover_mask_indices()

        # Baresoil value assignment
        emm[landcover_mask_indices["baresoil"]] = self.emissivity_soil_10
        # Vegetation value assignment
        emm[landcover_mask_indices["vegetation"]] = self.emissivity_veg_10
        # Mixed value assignment
        emm[landcover_mask_indices["mixed"]] = (0.004 * (
            ((self.ndvi[landcover_mask_indices["mixed"]] - 0.2) /
             (0.5 - 0.2))**2)) + 0.986
        return emm, emm


class ComputeEmissivityNBEM(Emissivity):
    """
    Method references:
    1. Li, Tianyu, and Qingmin Meng. "A mixture emissivity analysis method for
        urban land surface temperature retrieval from Landsat 8 data." Landscape
        and Urban Planning 179 (2018): 63-71.
    2. Yu, Xiaolei, Xulin Guo, and Zhaocong Wu. "Land surface temperature retrieval
        from Landsat 8 TIRS—Comparison between radiative transfer equation-based method,
        split window algorithm and single channel method." Remote sensing 6.10 (2014): 9829-9852.
    """

    emissivity_soil_10 = 0.9668
    emissivity_veg_10 = 0.9863
    emissivity_soil_11 = 0.9747
    emissivity_veg_11 = 0.9896

    def _compute_emissivity(self) -> np.ndarray:

        if self.red_band is None:
            raise ValueError(
                "Red band cannot be {} for this emissivity computation method".
                format(self.red_band))

        self.red_band = rescale_band(self.red_band)
        landcover_mask_indices = self._get_landcover_mask_indices()
        fractional_veg_cover = self._compute_fvc()

        def calc_emissivity_for_band(
            image,
            emissivity_veg,
            emissivity_soil,
            cavity_effect,
            red_band_coeff_a=None,
            red_band_coeff_b=None,
        ):
            image[landcover_mask_indices["baresoil"]] = red_band_coeff_a - (
                red_band_coeff_b *
                self.red_band[landcover_mask_indices["baresoil"]])

            image[landcover_mask_indices["mixed"]] = (
                (emissivity_veg *
                 fractional_veg_cover[landcover_mask_indices["mixed"]]) +
                (emissivity_soil *
                 (1 - fractional_veg_cover[landcover_mask_indices["mixed"]])) +
                cavity_effect[landcover_mask_indices["mixed"]])

            image[landcover_mask_indices["vegetation"]] = (
                emissivity_veg +
                cavity_effect[landcover_mask_indices["vegetation"]])
            return image

        emissivity_band_10 = np.empty_like(self.ndvi)
        emissivity_band_11 = np.empty_like(self.ndvi)
        frac_vegetation_cover = self._compute_fvc()

        cavity_effect_10 = cavity_effect(self.emissivity_veg_10,
                                         self.emissivity_soil_10,
                                         fractional_veg_cover)
        cavity_effect_11 = cavity_effect(self.emissivity_veg_11,
                                         self.emissivity_soil_11,
                                         fractional_veg_cover)

        emissivity_band_10 = calc_emissivity_for_band(
            emissivity_band_10,
            self.emissivity_veg_10,
            self.emissivity_soil_10,
            cavity_effect_10,
            red_band_coeff_a=0.973,
            red_band_coeff_b=0.047,
        )
        emissivity_band_11 = calc_emissivity_for_band(
            emissivity_band_11,
            self.emissivity_veg_11,
            self.emissivity_soil_11,
            cavity_effect_11,
            red_band_coeff_a=0.984,
            red_band_coeff_b=0.026,
        )
        return emissivity_band_10, emissivity_band_11


class ComputeEmissivityGopinadh(Emissivity):
    """
    Method reference:
    Rongali, Gopinadh, et al. "Split-window algorithm for retrieval of land surface temperature
    using Landsat 8 thermal infrared data." Journal of Geovisualization and Spatial Analysis 2.2
    (2018): 1-19.
    """

    emissivity_soil_10 = 0.971
    emissivity_veg_10 = 0.987

    emissivity_soil_11 = 0.977
    emissivity_veg_11 = 0.989

    def _compute_emissivity(self) -> np.ndarray:

        fractional_veg_cover = self._compute_fvc()

        def calc_emissivity_for_band(image, emissivity_veg, emissivity_soil,
                                     fractional_veg_cover):
            emm = (emissivity_soil *
                   (1 - fractional_veg_cover)) + (emissivity_veg *
                                                  fractional_veg_cover)
            return emm

        emissivity_band_10 = np.empty_like(self.ndvi)
        emissivity_band_10 = calc_emissivity_for_band(
            emissivity_band_10,
            self.emissivity_veg_10,
            self.emissivity_soil_10,
            fractional_veg_cover,
        )

        emissivity_band_11 = np.empty_like(self.ndvi)
        emissivity_band_11 = calc_emissivity_for_band(
            emissivity_band_11,
            self.emissivity_veg_11,
            self.emissivity_soil_11,
            fractional_veg_cover,
        )
        return emissivity_band_10, emissivity_band_11





# class NarrowBandEmissivity(Emissivity):

#     def __init__(self):
#         pass

#  class BroadbandEmissivity(Emissivity):
#     def __init__(self):
#         pass

# emissivity

# NDVI

# LC

# ASTER GED

# CMISS

# MMD related.


def transformX(beta, rel=None):
    # beta = emissivity / np.mean(emissivity)
    mmd_x = np.amax(beta) - np.amin(beta)
    mmr_x = np.log(np.amax(beta) / np.amin(beta))
    var_x = np.std(beta)
    X = np.vstack([mmd_x, mmr_x, var_x]).transpose()
    if rel == UniRel.MMD:
        return mmd_x
    elif rel == UniRel.MMR:
        return mmr_x
    elif rel == UniRel.VAR:
        return var_x
    return X


def get_sample_data(rsr, emis_lib, rel='Multi'):
    ''' get data for multivariate regression
	'''
    band_emissvity = get_band_emissivity(rsr, emis_lib)
    beta = RATIO(band_emissvity)
    mmd_x = np.amax(beta, axis=1) - np.amin(beta, axis=1)
    mmr_x = np.amax(beta, axis=1) / np.amin(beta, axis=1)
    var_x = np.std(beta, axis=1)
    X = np.vstack([mmd_x, mmr_x, var_x]).transpose()
    y = np.amin(band_emissvity, axis=1)

    if rel == UniRel.MMD:
        return mmd_x, y
    elif rel == UniRel.MMR:
        return np.log(mmr_x), y
    elif rel == UniRel.VAR:
        return var_x, y
    return X, y

    # def _load_data(self, emis_lib=None):
    # 	band_emissvity = get_band_emissivity(self.rsr, emis_lib)
    # 	y = np.amin(band_emissvity, axis=1)
    # 	beta = RATIO(band_emissvity)
    # 	if self.rel == 'mmd':
    # 		x = np.amax(beta, axis=1) - np.amin(beta, axis=1)
    # 	elif self.rel == 'mmr':
    # 		x = np.# LOG(np.amax(beta, axis=1) / np.amin(beta, axis=1))
    # 	elif self.rel == 'var':
    # 		x = np.std(beta, axis=1)
    # 	elif self.rel == 'mmd_# LOG':
    # 		x = np.# LOG(np.amax(beta, axis=1) - np.amin(beta, axis=1))
    # 	elif self.rel == 'mmr_# LOG':
    # 		x = np.# LOG(np.amax(beta, axis=1) / np.amin(beta, axis=1))
    # 	elif self.rel == 'var_# LOG':
    # 		x = np.# LOG(np.std(beta, axis=1))
    # 	else:
    # 		raise ValueError('unrecognized empirical relationship!')
    # 	return x, y


def _UNI_FUNC(val, a, b, c):
    ''' empirical function to be fitted
	'''
    return a + b * np.power(val, c)


class Relationship(object):
    ''' Univariate regression
	'''
    def __init__(self, rsr, emis_lib=DEFAULT_EMIS_LIB, rel=None):
        self.rsr = rsr
        self.emis_lib = emis_lib
        self.rel = rel
        self.y_pred = None

    def bias(self):
        return get_bias(self.y, self.y_pred)

    def rmse(self):
        return get_rmse(self.y, self.y_pred)

    def rsquare(self):
        return get_rsquare(self.y, self.y_pred)

    def validate(self, emis_lib=None, plot_fig=True):
        if emis_lib is None:
            emis_lib = DEFAULT_EMIS_LIB
        self._check()
        self.x, self.y = get_sample_data(self.rsr, emis_lib, self.rel)
        self.y_pred = self.func(self.x)
        if plot_fig:
            self.plot()
        print(f'rsquare:{self.rsquare()}\trmse:{self.rmse()}')

    def emis_func(self, beta):
        x = transformX(beta, self.rel)
        return self.func(x)

    def plot(self):
        self._check()
        plt.figure(figsize=[10, 8])
        min_x = np.min(self.y)
        max_y = np.max(self.y_pred)
        delta_y = (np.max(self.y_pred) - np.min(self.y_pred)) / 18
        plt.plot([min_x - 0.02, 1], [min_x - 0.02, 1], 'black', lw=0.8)
        plt.scatter(self.y, self.y_pred, c='r', marker='o', s=8)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        plt.text(min_x + 0.02, max_y - delta_y, '$N=%.f$' % len(self.y))
        plt.text(min_x + 0.02, max_y - delta_y * 2,
           '$R^2=%.5f$' % self.rsquare())
        # plt.text(min_x + 0.02, max_y - 0.06, '$BIAS=%.5f$' % self.bias())
        plt.text(min_x + 0.02, max_y - delta_y * 3,
           '$RMSE=%.5f$' % self.rmse())
        plt.title(f'{self.rel} fitting emis_min: predict and ground-truth')
        # plt.savefig(f'{self.rel}.eps', format='eps')
        plt.show()

    def _check(self):
        if self.y_pred is None:
            raise ValueError(
             f"{self.rel} relationship hasn't been constructed yet!")

    def __getstate__(self):
        return self.func

    def __setstate__(self, func):
        self.__init__(None)
        self.func = func


class UniRel(Relationship):
    ''' Univariate regression
	'''
    MMD = 'mmd'
    MMR = 'mmr'
    VAR = 'var'

    def __init__(self, rsr, rel=MMD, emis_lib=DEFAULT_EMIS_LIB):
        super().__init__(rsr, emis_lib, rel)

    def fit(self):
        self.x, self.y = get_sample_data(self.rsr, self.emis_lib, self.rel)
        self.opt, self.cov = optimize.curve_fit(_UNI_FUNC, self.x, self.y)
        a, b, c = self.opt
        self.func = partial(_UNI_FUNC, a=a, b=b, c=c)
        self.y_pred = self.func(self.x)
        return self.func

    def plot(self):
        self._check()
        plt.figure(figsize=[10, 8])
        a, b, c = self.opt
        left = np.min(self.x) - 0.1
        left = left if left > 0 else 0
        right = np.max(self.x) + 0.1
        new_x = np.power(self.x, c)
        plt.scatter(new_x, self.y, c='r', marker='o', s=8)
        x = np.power(np.linspace(left, right, num=100), c)
        y = list(map(lambda n: a + b * n, x))
        min_x = np.min(x)
        max_y = np.max(y)
        delta_y = (np.max(y) - np.min(y)) / 20
        delta_x = (np.max(x) - np.min(x)) / 3
        plt.plot(x, y)
        plt.xlabel('$%s^{%.4f}$' % (self.rel, c))
        plt.ylabel('emis_min')
        plt.text(min_x + delta_x, max_y - delta_y, '$N=%.f$' % len(self.y))
        plt.text(min_x + delta_x, max_y - delta_y * 2,
           '$R^2=%.5f$' % self.rsquare())
        # plt.text(max_x - 0.3, max_y - 0.06, '$BIAS=%.5f$' % self.bias())
        plt.text(min_x + delta_x, max_y - delta_y * 3,
           '$RMSE=%.5f$' % self.rmse())
        plt.text(min_x + delta_x, max_y - delta_y * 4,
           '$emis=%.4f%.4f%s^{%.5f}$' % (a, b, self.rel, c))
        plt.title(f'{self.rel} fitting: emis_min and {self.rel}')
        # plt.savefig(f'{self.rel}.eps', format='eps')
        plt.show()


class MultiRel(Relationship):
    ''' Multivariate regression
	'''
    MLR = 'mlr'
    BRR = 'brr'
    SVR = 'svr'
    GBR = 'gbr'

    def __init__(self, rsr=None, rel=GBR, emis_lib=DEFAULT_EMIS_LIB):
        if rsr != None:
            super().__init__(rsr, emis_lib, rel)
            self._load_model()

    def score(self, fold=10):
        return cross_val_score(self.model, self.x, self.y, cv=fold)

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, filename):
        self.model = pickle.load(open(filename, 'rb'))
        self.func = self.model.predict

    def metric(self):
        model_metrics_name = [
         explained_variance_score, mean_absolute_error, mean_squared_error,
         r2_score
        ]
        metric = []
        for m in model_metrics_name:
            score = m(self.y, self.y_pred)
            metric.append(score)
        print('short name \t full name')
        print('ev\texplained_variance')
        print('mae\tmean_absolute_error')
        print('mse\tmean_squared_error')
        print('r2\tr square')
        print(metric)

    def fit(self):
        self.x, self.y = get_sample_data(self.rsr, self.emis_lib, self.rel)
        self.y_pred = self.model.fit(self.x, self.y).predict(self.x)
        self.func = self.model.predict

    def _load_model(self):
        if self.rel == MultiRel.MLR:
            self.model = LinearRegression()
        elif self.rel == MultiRel.BRR:
            self.model = BayesianRidge()
        elif self.rel == MultiRel.SVR:
            self.model = SVR()
        elif self.rel == MultiRel.GBR:
            self.model = GradientBoostingRegressor()
        else:
            self.model = LinearRegression()


class MultiRelComp(object):
    ''' Intercomparison btween varieties of empirical relationships
	'''
    def __init__(self, rsr, emis_lib=DEFAULT_EMIS_LIB):
        self.emis_lib = emis_lib
        self.regression = None
        self.rsr = rsr

    def fit(self, model_name='models.ckpt'):
        X, y = get_sample_data(self.rsr, self.emis_lib)
        # LOG.info('regression prediction')
        self.regression = Regression(X, y)
        self.regression.add_models(
         BayesianRidge=BayesianRidge(),
         LinearRegression=LinearRegression(),
         SVR=SVR(gamma='scale'),
         GradientBoostingRegressor=GradientBoostingRegressor())
        self.regression.evaluate()
        self.regression.print()
        self.regression.plot()
        self.regression.save(model_name)

    def validate(self, emis_lib=None):
        if emis_lib is None:
            emis_lib = DEFAULT_EMIS_LIB
        X_test, y_test = get_sample_data(self.rsr, emis_lib)
        self.regression.validate(X_test, y_test)

    def load(self, model_name):
        self.regression = Regression()
        self.regression.load(model_name)
        self.regression.plot()


class Regression:
    ''' regression model
	'''
    def __init__(self, X=None, y=None, fold=10):
        self.X = X
        self.y = y
        self.fold = fold
        self.models = defaultdict()
        self.pre_y = defaultdict()

    def load(self, filename):
        regression = pickle.load(open(filename, 'rb'))
        self.X = regression['X']
        self.y = regression['y']
        self.models = regression['models']
        self.evaluate()

    def save(self, filename):
        regression = defaultdict()
        regression['X'] = self.X
        regression['y'] = self.y
        regression['models'] = self.models
        pickle.dump(regression, open(filename, 'wb'))

    def plot(self):
        n_samples, n_features = self.X.shape
        plt.figure(figsize=[10, 8])
        left = np.min(self.y)
        right = np.max(self.y)
        delta = (right - left) / 20
        new_y = np.linspace(left - delta, right + delta, 100)
        plt.plot(new_y, new_y, color='k', label='true y')

        cmap = get_cmap(len(self.models)).reversed()
        color_lst = ['g', 'm', 'c', 'r', 'y', 'b', 'k', 'w']
        filled_markers = ('o', 'v', '8', 's', 'p', '*', 'h', '^', 'H', 'D',
              '>', 'd', 'P', 'X', '<')

        for i, model_name in enumerate(self.models.keys()):
            plt.scatter(self.y,
               self.pre_y[model_name],
               color=color_lst[i],
               marker=filled_markers[i],
               label=model_name)
        plt.title('Comprison of Multivariate Regressions')
        plt.legend(loc='upper left')
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        plt.show()

    def evaluate(self):
        model_metrics_name = [
         explained_variance_score, mean_absolute_error, mean_squared_error,
         r2_score
        ]
        cv_scores = []
        model_metrics = []
        model_names = []
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X, self.y, cv=self.fold)
            cv_scores.append(scores)
            self.pre_y[name] = model.fit(self.X, self.y).predict(self.X)
            metric = []
            for m in model_metrics_name:
                score = m(self.y, self.pre_y[name])
                metric.append(score)
            metric.append(math.sqrt(metric[-2]))
            model_metrics.append(metric)
            model_names.append(name)
        self.cv_df = pd.DataFrame(cv_scores, index=model_names)
        self.re_df = pd.DataFrame(model_metrics,
                index=model_names,
                columns=['ev', 'mae', 'mse', 'r2', 'rmse'])

    def print(self):
        n_samples, n_features = self.X.shape
        print('samples: %d \t features: %d' % (n_samples, n_features))
        print(70 * '-')
        print('cross validation result:')
        print(self.cv_df)
        print(70 * '-')
        print('regression metrics:')
        print(self.re_df)
        print(70 * '-')
        print('short name \t full name')
        print('ev\texplained_variance')
        print('mae\tmean_absolute_error')
        print('mse\tmean_squared_error')
        print('r2\tr square')
        print(70 * '-')  # 打印分隔线

    def add_models(self, **kargs):
        for k, v in kargs.items():
            self.models[k] = v

    def validate(self, X, y):
        results = defaultdict()
        model_names = []
        scores = []
        for name, model in self.models.items():
            model_names.append(name)
            results[name] = model.predict(X)
            scores.append(
             (r2_score(y, results[name]), get_rsquare(y, results[name]),
              get_rmse(y, results[name])))
        pre_df = pd.DataFrame(scores,
               index=model_names,
               columns=['r2', 'r-square', 'rmse'
                  ])  # columns=['ev', 'mae', 'mse', 'r2']
        print('validation result r square:')
        print(70 * '-')
        print(pre_df)
        return results

    def predict(self, X):
        results = defaultdict()
        for name, model in self.models.items():
            results[name] = model.predict(X)
        return results
