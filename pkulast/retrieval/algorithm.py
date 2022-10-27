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
Raster Data Pixel-Level Temperature and  Emissivity 
Separation Algorithm Demo(Concurrent Version)
"""


import sys
import logging
import numpy as np
from scipy.interpolate import interp1d
from collections.abc import Iterable
from pkulast.utils.collections import *
from pkulast.config import ALG_CFG, DEFAULT_EMIS_LIB, SW_ALG, SC_ALG
from pkulast.utils.algorithm import parse_formula
from pkulast.retrieval.coefficient import sw_coefficient, sc_coefficient
# from numba import int32, float32, string    # import the types
# from numba.experimental import jitclass

# spec = [
# 	('lst', float32),
# 	('emission', float32[:]),
#     ('QA', string),               # a simple scalar field
#     ('land_rad', float32[:]),          # an array field
# 	('land_rad', float32[:]),
# 	('down_rad', float32[:]),
# 	('land_rad', float32[:]),
# 	('land_rad', float32[:]),
# ]

sys.setrecursionlimit(1500)

LOG = logging.getLogger(__name__)


# @jitclass(spec)
class TES:
    """
	TES Algorithm main body
	"""
    def __init__(self, lut, emis_func=None):
        self.lst = 0.0  # TES LST output
        self.emission = np.NaN  # TES LSE output
        self.QA = ""  # quality assurance
        self.land_rad = np.NaN  # land leaving radiance
        self.down_rad = np.NaN  # atmosphere down-welling radiance
        self.lut = LookupTable(lut)  # LookupTable(lut)
        self.emis_func = emis_func

        # TES algorithm relevant coefficients
        self.t1 = 0.05  # Div normally equals to NEDT (noise equivalent differential temperature)
        self.t2 = 0.05  # Con normally equals to NEDT (noise equivalent differential temperature)
        self.iter_num = 12  # number of iterations for NEM module
        self.emission_reset = False  # emissivity not reset yet
        self.V1 = 0.00017  # variance of emission
        # self.V1 = 0.015
        self.V2 = 0.001
        self.V3 = 0.001
        self.V4 = 0.0001
        self.refined_value = 0.96
        self.m_value = 0.903

    def compute(self, landrad, downrad, initial_value=0.99):
        self.land_rad = np.asarray(landrad)  # land leaving radiance
        self.down_rad = np.asarray(downrad)  # downwelling radiance
        self.emission = np.empty_like(landrad)
        self.initial_value = initial_value
        self.tes()
        # LOG.info(self.lst)
        return np.hstack((self.emission, np.array([
            self.lst,
        ])))

    def simulate(self, lst, emis, downrad, tau, error, error2):
        # LOG.info('****************************************************')
        # LOG.info("LST:" + str(lst))
        # LOG.info('****************************************************')
        # LOG.info("MMD:" + str(get_mmd(emis)))
        # LOG.info('****************************************************')
        beta = self.RATIO(emis)
        # LOG.info("DEL:" + str(self.emis_func(beta) - np.min(emis)))
        # LOG.info('****************************************************')
        land_rad = emis * self.lut.bt2r(lst) + (1 - emis) * downrad
        land_rad += (self.lut.bt2r(lst + np.random.normal(0, error)) -
                     self.lut.bt2r(lst)) / tau
        initial_value = float(np.max(emis)) + np.random.normal(0, error2)
        self.refined_value = initial_value
        self.m_value = float(np.min(emis)) + np.random.normal(0, error2)
        return self.compute(land_rad, downrad, initial_value)

    def __getstate__(self):
        return self.n

    def __setstate__(self, n):
        self.__init__(n)

    def tes(self):
        """
		TES algorithm body
		:return: void
		"""
        exec_success = self.NEM(
            self.initial_value)  # NEM module execute successfully or not
        if exec_success:  # initial emission_max success
            variance = np.var(self.emission)  # variance of emission
            # LOG.info("VARIANCE: "+ str(variance))
            if variance > self.V1:  # rock etc. non-grey body
                # LOG.info("VARIANCE > THRESHOLD: "+ f"{self.lst}, {self.emission}")
                exec_success = self.NEM(self.refined_value)
                if exec_success:
                    # LOG.info("VARIANCE > THRESHOLD SUCCESS: "+ f"{self.lst}, {self.emission}")
                    beta = self.RATIO(self.emission)
                    self.MMD(beta)
                    # # LOG.info("MMD: ", self.lst, self.emission)
                else:
                    self.QA = "bad data!"
                    # LOG.info(self.QA)
            else:  #  grey body refine emission_max
                # LOG.info("VARIANCE < THRESHOLD: "+ f"{self.lst}, {self.emission}")
                emission = self.emission
                lst = self.lst
                emission_refined = self.refine_emission_max(variance)
                # LOG.info("REFINED EMISSION before: " + f"{self.lst}, {self.emission}")
                # LOG.info("Emission Refined:" + str(emission_refined))
                # LOG.info(self.QA)
                if not emission_refined:  # emission_max not refined
                    # exec_success = self.NEM(0.983)
                    self.emission = emission
                    self.lst = lst
                else:
                    pass
                    # LOG.info("REFINED EMISSION after: ", self.lst, self.emission)
                beta = self.RATIO(self.emission)
                if self.MMD(beta):
                    self.QA = "success"
                    # LOG.info("MMD: " + f":   {self.lst} {self.emission.tolist()}")
                else:
                    self.QA = "bad data!"
                    # LOG.info(self.QA)
        else:
            self.QA = "bad data!"
            # LOG.info(self.QA)

    def NEM(self, emission_max):
        """
		:param emission_max: 最大发射率
		:return: boolean NEM模块执行是否成功
		"""
        R_est = np.zeros((self.iter_num, self.land_rad.shape[0]))

        R_est[0, :] = self.land_rad - (1 - emission_max) * self.down_rad
        lst_eval = self.lut.r2bt(R_est[0, :] / emission_max)

        self.lst = np.max(lst_eval)
        self.emission = R_est[0, :] / self.lut.bt2r(self.lst)

        # # LOG.info("First   ", self.lst, self.emission)
        for i in range(1, self.iter_num):
            # # LOG.info(emission_max, self.emission)
            R_est[i, :] = self.land_rad - (1 - self.emission) * self.down_rad
            lst_eval = self.lut.r2bt(R_est[i, :] / self.emission)
            # LOG.info("Mid result:   " + str(lst_eval))
            self.lst = np.max(lst_eval)
            self.emission = R_est[i, :] / self.lut.bt2r(self.lst)
            # LOG.info("MMD:" + str(get_mmd(self.emission)))
            # LOG.info("NEM: Time " + str(i) + ":   " + f"{self.lst}, {self.emission}")
            # LOG.info('Test2 ' + str(np.max(np.abs(R_est[i, :] - R_est[i - 1, :]))))
            con = np.max(np.abs(R_est[i, :] - R_est[i - 1, :])) < self.t2
            if con:
                self.emission = (self.land_rad - self.down_rad) / (
                    self.lut.bt2r(self.lst) - self.down_rad)
                # LOG.info("Iteration Success:   " + f"{self.lst}, {self.emission}")
                return True
            if i > 1:
                div = np.min(
                    np.abs(R_est[i, :] + R_est[i - 2, :] -
                           2.0 * R_est[i - 1, :])) > self.t1
                if div:
                    # LOG.info("Iteration Failed:   "+ f"{self.lst}, {self.emission}")
                    return False
                #self.NEM(np.max(self.emission))
        # 循环12次仍然不收敛
        return False

    def refine_emission_max(self, v):
        """
		:param v: variance of self.emission
		:return:boolean 模块执行是否成功
		"""
        # LOG.info("Refine Emission-Max Execution")
        E = [0.99, 0.97, 0.95, 0.92]
        V = [v]
        for emission in [0.97, 0.95, 0.92]:
            exec_success = self.NEM(emission)
            if exec_success:
                V.append(np.var(self.emission))
            else:
                return False
        a, b, c = np.polyfit(E, V, 2)
        minv_emission = -b / (2.0 * a)
        if 0.9 < minv_emission < 1.0:
            # vCon = np.mean([np.abs((V[2] - V[1]) / (E[2] - E[1])),
            #       np.abs((V[3] - V[2]) / (E[3] - E[2]))])
            vCon = 2 * a * minv_emission + b
            if vCon < self.V2:
                # vDiv = np.abs((V[3] + V[1] - 2.0 * V[2]) / 0.01)
                vDiv = 2 * a
                if vDiv > self.V3:
                    minV = np.polyval((a, b, c), minv_emission)
                    if minV < self.V4:
                        exec_success = self.NEM(minv_emission)
                        # LOG.info("testV1", exec_success, self.lst, self.emission)
                        return exec_success
                    else:
                        self.QA = "testV4 failed"
                        # LOG.info(self.QA + f":   {self.lst} {self.emission.tolist()}")
                        return False
                else:
                    self.QA = "testV3 failed"
                    # LOG.info(self.QA + f":   {self.lst} {self.emission.tolist()}")
                    return False
            else:
                self.QA = "testV2 failed"
                # LOG.info(self.QA + f":   {self.lst} {self.emission.tolist()}")
                return False  # 失败直接进入RATIO模块
        else:
            self.QA = "emission-max not in range(0.9, 1.0)"
            # LOG.info(self.QA + f":   {self.lst} {self.emission.tolist()}")
            return False

    @staticmethod
    def RATIO(emission):
        return emission / np.mean(emission)

    def MMD(self, beta):
        M = 1  # iteration times
        for j in range(M):
            delta = np.max(beta) - np.min(beta)
            # LOG.info("MMD:" + str(delta))
            if delta < 0.006:
                emission_min = 0.983
            else:
                # MMD' = [MMD^2 - cNEDE^2]^(-1) c=1.52, NEDE=0.0032, 0.3K@300K
                # emission_min = 0.9864 - 0.7865 * (delta ** 0.8529)
                # − 0.687 0.737 0.994
                # emission_min = 0.99 - 0.7334 * (delta ** 0.7973)
                # LOG.info("Beta:" + str(beta))
                beta = beta.astype(np.float32)
                # delta = np.amax(beta) - np.amin(beta)
                # try:
                emission_min = self.emis_func(beta)
                # emission_min = 0.994 - 0.687 * (delta**0.737)
                # except:
                # 	emission_min = self.m_value
            self.emission = beta * emission_min / (np.min(beta))
            # LOG.info("MMD:" + f"{get_mmd(self.emission).tolist()}")
            R = self.land_rad - (1 - self.emission) * self.down_rad
            T = self.lut.r2bt(R / self.emission)
            # LOG.info(f'LST STD: {np.std(T)}')
            # emission_max = np.max(self.emission)
            # R = self.land_rad - (1 - emission_max) * self.down_rad
            # T = self.lut.r2bt(R / emission_max)
            # LOG.info("MMD: Time " + str(j) + f":   {self.lst} {self.emission.tolist()}")
            self.lst = np.max(T)
            self.emission = (self.land_rad - self.down_rad) / (
                self.lut.bt2r(self.lst) - self.down_rad)
            beta = self.RATIO(self.emission)
        return True


class TESv2:
    def __init__(self, lut, emis_func=None):
        self.lst = 0.0  # TES LST output
        self.emission = np.NaN  # TES LSE output
        self.QA = ""  # quality assurance
        self.land_rad = np.NaN  # land leaving radiance
        self.down_rad = np.NaN  # atmosphere down-welling radiance
        self.lut = LookupTable(lut)  # LookupTable(lut)
        self.emis_func = emis_func
        # TES algorithm relevant coefficients
        self.t1 = 0.05  # 0.05  # Div normally equals to NEDT (noise equivalent differential temperature)
        self.t2 = 0.05  # 0.05  # Con normally equals to NEDT (noise equivalent differential temperature)
        self.MMD_iter_num = 1
        self.MMD_threshold = 0.006  # 0.032
        self.V1 = 0.00017  # 0.00017  # variance of emission
        self.V2 = 0.001  #0.001
        self.V3 = 0.001  #0.001
        self.V4 = 0.0001  # 0.0001
        self.emis_max = 0.99
        self.emis_refined = 0.96
        self.emis_threshold = 0.015
        self.emis_class = {'greybody': 0.983, 'non-greybody': 0.967}

    def __getstate__(self):
        return self.n

    def __setstate__(self, n):
        self.__init__(n)

    def compute(self, landrad, downrad, initial_value=0.99):
        self.land_rad = np.asarray(landrad)  # land leaving radiance
        self.down_rad = np.asarray(downrad)  # downwelling radiance
        self.emission = np.empty_like(landrad)
        self.initial_value = initial_value
        self.tes()
        # LOG.info(self.lst)
        return np.hstack((self.emission, np.array([
            self.lst,
        ])))

    def tes(self):
        exec_success = self.NEM(
            self.emis_max)  # NEM module execute successfully or not
        if exec_success:  # initial emission_max success
            variance = np.var(self.emission)  # variance of emission
            if variance > self.V1:  # water rock etc. grey body
                exec_success = self.NEM(self.emis_class['non-greybody'])
                if exec_success:
                    # print(self.emission)
                    beta = self.RATIO(self.emission)
                    self.MMD(beta)
                else:
                    self.QA = "bad data!"
            else:  # grey body refine emission_max
                exec_success = self.NEM(self.emis_class['greybody'])
                if exec_success:
                    beta = self.RATIO(self.emission)
                    self.MMD(beta)
                else:
                    self.QA = "bad data!"
        else:
            self.QA = "bad data!"

    def NEM(self, emission_max):
        R_est = self.land_rad - (1 - emission_max) * self.down_rad
        lst_eval = self.lut.r2bt(R_est / emission_max)
        self.lst = np.max(lst_eval)
        sorted_lst = np.sort(lst_eval)
        if sorted_lst[-1] - sorted_lst[-2] >= 5:
            pass
            print("warning in BT!")
        self.emission = R_est / self.lut.bt2r(self.lst)
        return True

    @staticmethod
    def RATIO(emission):
        return emission / np.mean(emission)

    def MMD(self, beta):
        M = 1  # iteration times
        for _ in range(M):
            delta = np.max(beta) - np.min(beta)
            if delta < self.MMD_threshold:
                emission_min = self.emis_class['greybody']  #0.982
            else:
                emission_min = self.emis_func(beta)
                # delta = np.amax(beta) - np.amin(beta)
                # emission_min = 0.994 - 0.687 * (delta**0.737)
            self.emission = beta * emission_min / (np.min(beta))
            R = self.land_rad - (1 - self.emission) * self.down_rad
            emission_max = np.max(self.emission)
            T = self.lut.r2bt(R / emission_max)
            self.lst = np.max(T)
            self.emission = (self.land_rad - self.down_rad) / (
                self.lut.bt2r(self.lst) - self.down_rad)
            beta = self.RATIO(self.emission)
        return True


class TESv3:
    def __init__(self, lut, emis_func=None):
        self.lst = 0.0  # TES LST output
        self.emission = np.NaN  # TES LSE output
        self.QA = ""  # quality assurance
        self.land_rad = np.NaN  # land leaving radiance
        self.down_rad = np.NaN  # atmosphere down-welling radiance
        self.lut = LookupTable(lut)  # LookupTable(lut)
        self.emis_func = emis_func
        # TES algorithm relevant coefficients
        self.t1 = 0.05  # 0.05  # Div normally equals to NEDT (noise equivalent differential temperature)
        self.t2 = 0.05  # 0.05  # Con normally equals to NEDT (noise equivalent differential temperature)
        self.MMD_iter_num = 1
        self.MMD_threshold = 0.006  # 0.032
        self.V1 = 0.00017  # 0.00017  # variance of emission
        self.V2 = 0.001  #0.001
        self.V3 = 0.001  #0.001
        self.V4 = 0.0001  # 0.0001
        self.emis_max = 0.99
        self.emis_refined = 0.96
        self.emis_threshold = 0.015
        self.emis_class = {'greybody': 0.983, 'non-greybody': 0.967}

    def __getstate__(self):
        return self.n

    def __setstate__(self, n):
        self.__init__(n)

    def compute(self, landrad, downrad, initial_value=0.99):
        self.land_rad = np.asarray(landrad)  # land leaving radiance
        self.down_rad = np.asarray(downrad)  # downwelling radiance
        self.emission = np.empty_like(landrad)
        self.initial_value = initial_value
        self.tes()
        return np.hstack((self.emission, np.array([
            self.lst,
        ])))

    def tes(self):
        exec_success = self.NEM(
            self.emis_max)  # NEM module execute successfully or not
        if exec_success:  # initial emission_max success
            variance = np.var(self.emission)  # variance of emission
            if variance > self.V1:  # water rock etc. grey body
                exec_success = self.NEM(self.emis_class['non-greybody'])
                if exec_success:
                    # print(self.emission)
                    beta = self.RATIO(self.emission)
                    self.MMD(beta)
                else:
                    self.QA = "bad data!"
            else:  # grey body refine emission_max
                exec_success = self.NEM(self.emis_class['greybody'])
                if exec_success:
                    beta = self.RATIO(self.emission)
                    self.MMD(beta)
                else:
                    self.QA = "bad data!"
        else:
            self.QA = "bad data!"

    def NEM(self, emission_max):
        R_est = self.land_rad - (1 - emission_max) * self.down_rad
        lst_eval = self.lut.r2bt(R_est / emission_max)
        self.lst = np.max(lst_eval)
        sorted_lst = np.sort(lst_eval)
        if sorted_lst[-1] - sorted_lst[-2] >= 5:
            print("warning in BT!")
        self.emission = R_est / self.lut.bt2r(self.lst)
        return True

    @staticmethod
    def RATIO(emission):
        return emission / np.mean(emission)

    def MMD(self, beta):
        M = 1  # iteration times
        for _ in range(M):
            delta = np.max(beta) - np.min(beta)
            # if delta < self.MMD_threshold:
            #     emission_min = self.emis_class['greybody']  #0.982
            # else:
            emission_min = self.emis_func(beta)
            # delta = np.amax(beta) - np.amin(beta)
            # emission_min = 0.955 - 0.863 * (delta**1.000)
            self.emission = beta * emission_min / (np.min(beta))
            R = self.land_rad - (1 - self.emission) * self.down_rad
            emission_max = np.max(self.emission)
            T = self.lut.r2bt(R / emission_max)
            self.lst = np.max(T)
            self.emission = (self.land_rad - self.down_rad) / (
                self.lut.bt2r(self.lst) - self.down_rad)
            beta = self.RATIO(self.emission)
        return True


class WVSTES:
    def __init__(self):
        pass


paras_spec = {
    'tau': "transmittance",
    'w': "water vapor",
    'Ta': 'effective  atmospheric temperature',
    'Lu': "upward radiance",
    'Ld': "downward radiance",
    'e': "emissivity",
    'Ts': "surface temperature",
    'T0': "kelvin temperture at 0℃",
    'mu': "cosine of viewing angle",
}


class SC:
    _algs = ALG_CFG['SC']
    algorithms = list(_algs.keys())

    def __init__(self, algorithm="RTE"):
        self.alg = algorithm
        self.req = self._algs[self.alg]['requirements']
        self.formula = self._algs[self.alg]['formula']
        self.pre_expr = self._algs[self.alg].get("pre_expr")
        self.post_expr = self._algs[self.alg].get("post_expr")
        self.optimized_target = self._algs[self.alg].get("optimized_target")
        self.optimized_params = self._algs[self.alg].get("optimized_params")
        self.coef_count = 0
        for target in self.optimized_target:
            self.coef_count += len(target) + 1 if self.optimized_params.get("fit_intercept") else len(target)
        self.params_filled = False
        self.models = []


    def get_models(self):
        if not self.models:
            if not self.params_filled:
                raise ValueError("Please set parameters first")
            self.models = sc_coefficient(
                self.rsr, self.alg, self.dTs, self.emis_lib, self.profile_lib, self.cwv_intervals, self.mu)
        return self.models

    def add_models(self, models):
        self.models.extend(models)

    def add_coeffs(self, coeffs, label="user defined coeffs"):
        self.models.append({
            "label": label,
            "coeffs": coeffs,
            "residuals": None,
            "p_LST": None,
            "t_LST": None,
        })

    def set_params(self,
                   rsr,
                   dTs=range(-10, 25, 5),
                   emis_lib=DEFAULT_EMIS_LIB,
                   profile_lib="TIGR946",
                   cwv_intervals=[[0.0, 2.5], [2.0, 3.5], [3.0, 4.5],
                                  [4.0, 5.5], [5.0, 6.3]],
                   bt_intervals=None,
                   mu=[
                       1,
                   ]):
        self.rsr = rsr
        self.dTs = dTs
        self.emis_lib = emis_lib
        self.profile_lib = profile_lib
        self.cwv_intervals = cwv_intervals
        self.bt_intervals = bt_intervals
        self.mu = mu
    def get_params(self):
        pass

    def _construct_formula(self, kwargs):
        sw_params = []
        kwargs['constant'] = 0
        if self.pre_expr:
            for k in self.pre_expr.keys():
                kwargs[k] = parse_formula(self.pre_expr[k], kwargs)
        kwargs['constant'] = np.squeeze(kwargs['constant'])
        params = {}
        for k, forms in self.optimized_target.items():
            sw_params = []
            for f_str in forms:
                sw_params.append(parse_formula(f_str, kwargs))
            params[k] = np.concatenate(
                sw_params,
                axis=1,
            )
        if self.post_expr:
            for k in self.post_expr.keys():
                kwargs[k] = parse_formula(self.post_expr[k], kwargs)
        return params, kwargs

    def sc(self, params, kwargs, model_index=0, indices=None):
        coeffs = self.models[model_index]["coeffs"]
        for i, k in enumerate(self.optimized_target.keys()):
            intercept_, coef_ = coeffs[i][0], coeffs[i][1:]
            param = params[k] if indices is None else params[k][indices]
            predict_key = 'p' + k[1:]
            target = np.matmul(param, coef_) + intercept_ + kwargs['constant']
            kwargs[predict_key] = target.reshape(-1, 1)
        if self.post_expr:
            for k in self.post_expr.keys():
                kwargs[k] = parse_formula(self.post_expr[k], kwargs)
        res = parse_formula(self.formula, kwargs)
        return res

    def compute(self, bt, model_index=0, **kwargs):
        for req in self.req:
            if req not in kwargs.keys():
                raise ValueError(f"missing required parameter {req}")
        emis = kwargs['e'] if 'e' in kwargs.keys() else np.ones_like(bt)
        rad = kwargs['L'] if 'L' in kwargs.keys() else np.ones_like(bt)
        tcwv = kwargs['w'] if 'w' in kwargs.keys() else np.zeros(len(bt))
        mu = kwargs['mu'] if 'mu' in kwargs.keys() else np.ones(len(bt))
        tau = kwargs['tau'] if 'tau' in kwargs.keys() else np.ones(len(bt))
        Lu = kwargs['Lu'] if 'Lu' in kwargs.keys() else np.ones(len(bt))
        Ld = kwargs['Ld'] if 'Ld' in kwargs.keys() else np.ones(len(bt))
        params_ab = self.rsr.derivation_ab()
        kwargs.update({
                    "e": emis.reshape(-1, 1),
                    "T0": 273.15,
                    "T": bt.reshape(-1, 1),
                    "w": tcwv.reshape(-1, 1),
                    "L": rad.reshape(-1, 1),
                    "mu": mu.reshape(-1, 1),
                    "tau":tau.reshape(-1, 1),
                    "Lu":Lu.reshape(-1, 1),
                    "Ld":Ld.reshape(-1, 1),
                    "a": params_ab[0][0],
                    "b": params_ab[0][1],
                    "blambda": self.rsr.b_lambda()[0],
                })
        params, kwargs = self._construct_formula(kwargs)
        result = self.sc(params, kwargs, model_index=model_index)
        # if 'w' in self.req:
        #     result = self.sc(params, kwargs, model_index=0)
        # else:
        #     if 'w' in kwargs.keys():# grouped by cwv
        #         w = kwargs.get('w')
        #         for i, interval in enumerate(self.cwv_intervals):
        #             l, r = interval
        #             indices = np.where((w >= l) & (w < r))
        #             result[indices] = self.sc(params, kwargs, model_index=i, indices=indices)
        #     else: # no group
        #         result = result = self.sc(params, kwargs, model_index=0)
        return result

class SW:
    _algs = ALG_CFG['SW']
    algorithms = list(_algs.keys())

    def __init__(self, algorithm='Wan14'):
        self.alg = algorithm
        self.req = self._algs[self.alg]['requirements']
        self.formula = self._algs[self.alg]['formula']
        self.pre_expr = self._algs[self.alg].get("pre_expr")
        self.post_expr = self._algs[self.alg].get("post_expr")
        self.optimized_target = self._algs[self.alg].get("optimized_target")
        self.optimized_params = self._algs[self.alg].get("optimized_params")
        self.coef_count = 0
        for target in self.optimized_target:
            self.coef_count += len(target) + 1 if self.optimized_params.get("fit_intercept") else len(target)
        self.params_filled = False
        self.models = []

    def get_models(self):
        if not self.models:
            if not self.params_filled:
                raise ValueError("Please set parameters first")
            self.models = sw_coefficient(
                self.rsr, self.alg, self.dTs, self.emis_lib, self.profile_lib, self.cwv_intervals, self.mu)
        return self.models

    def add_models(self, models):
        self.models.extend(models)

    def add_coeffs(self, coeffs, label="user defined coeffs"):
        self.models.append({
            "label": label,
            "coeffs": coeffs,
            "residuals": None,
            "p_LST": None,
            "t_LST": None,
        })


    def set_params(self,
                   rsr,
                   dTs=range(-10, 25, 5),
                   emis_lib=DEFAULT_EMIS_LIB,
                   profile_lib="TIGR946",
                   cwv_intervals=[[0.0, 2.5], [2.0, 3.5], [3.0, 4.5],
                                  [4.0, 5.5], [5.0, 6.3]],
                   bt_intervals=None,
                   mu=[
                       1,
                   ]):
        self.rsr = rsr
        self.dTs = dTs
        self.emis_lib = emis_lib
        self.profile_lib = profile_lib
        self.cwv_intervals = cwv_intervals
        self.bt_intervals = bt_intervals
        self.mu = mu
        self.params_filled = True

    def get_params(self):
        pass

    def _construct_formula(self, kwargs):
        sw_params = []
        kwargs['constant'] = 0
        if self.pre_expr:
            for k in self.pre_expr.keys():
                kwargs[k] = parse_formula(self.pre_expr[k], kwargs)
        kwargs['constant'] = np.squeeze(kwargs['constant'])
        params = {}
        for k, forms in self.optimized_target.items():
            sw_params = []
            for f_str in forms:
                sw_params.append(parse_formula(f_str, kwargs))
            params[k] = np.concatenate(
                sw_params,
                axis=1,
            )
        return params, kwargs

    def sw(self, params, kwargs, model_index=0, indices=None):
        coeffs = self.models[model_index]["coeffs"]
        for i, k in enumerate(self.optimized_target.keys()):
            intercept_, coef_ = coeffs[i][0], coeffs[i][1:]
            param = params[k] if indices is None else params[k][indices]
            predict_key = 'p' + k[1:]
            target = np.matmul(param, coef_) + intercept_ + kwargs['constant']
            kwargs[predict_key] = target.reshape(-1, 1)
        if self.post_expr:
            for k in self.post_expr.keys():
                kwargs[k] = parse_formula(self.post_expr[k], kwargs)
        res = parse_formula(self.formula, kwargs)
        return res

    def compute(self, bt, model_index=0, **kwargs):
        for req in self.req:
            if req not in kwargs.keys():
                raise ValueError(f"missing required parameter {req}")
        emis = kwargs['e'] if 'e' in kwargs.keys() else np.ones_like(bt)
        tcwv = kwargs['w'] if 'w' in kwargs.keys() else np.zeros(len(bt))
        mu = kwargs['mu'] if 'mu' in kwargs.keys() else np.ones(len(bt))
        kwargs.update({
                    "e": ((emis[:, 0] + emis[:, 1]) / 2.0).reshape(-1, 1),
                    "de": ((emis[:, 0] - emis[:, 1])).reshape(-1, 1),
                    "e11": emis[:, 0].reshape(-1, 1),
                    "e12": emis[:, 1].reshape(-1, 1),
                    "T0": 273.15,
                    "T11": bt[:, 0].reshape(-1, 1),
                    "T12": bt[:, 1].reshape(-1, 1),
                    "dT": ((bt[:, 0] - bt[:, 1])).reshape(-1, 1),
                    "w": tcwv.reshape(-1, 1),
                    "mu": mu.reshape(-1, 1),
                })
        params, kwargs = self._construct_formula(kwargs)
        result = self.sw(params, kwargs, model_index=model_index)
        # if 'w' in self.req:
        #     result = self.sw(params, kwargs, model_index=0)
        # else:
        #     if 'w' in kwargs.keys():# grouped by cwv
        #         w = kwargs.get('w')
        #         for i, interval in enumerate(self.cwv_intervals):
        #             l, r = interval
        #             indices = np.where((w >= l) & (w < r))
        #             result[indices] = self.sw(params, kwargs, model_index=i, indices=indices)
        #     else: # no group
        #         result = self.sw(params, kwargs, model_index=0)
        return result


class LookupTable:
    """Look-up Table implementation
	"""
    def __init__(self, data, selected_band=None):
        self.data = data
        self.rows, self.cols = data.shape
        selected_band = selected_band if selected_band else list(
            range(self.cols - 1))
        self.selected_band = np.array(selected_band)
        self.band_count = self.selected_band.size
        self.selected_col = 0

    def bi_search(self, l, r, val):
        """ 二分搜索
		"""
        if r - l <= 1:
            return l, r
        mid = int((r + l) / 2)
        mid_val = self.data[mid][self.selected_col]
        if mid_val > val:
            return self.bi_search(l, mid, val)
        elif mid_val < val:
            return self.bi_search(mid, r, val)
        else:
            return mid, mid

    def bt2r(self, T):
        """
		根据LST返回各波段黑体辐射值
		"""
        self.selected_col = 0
        if self.boundary_check(T):
            return self.interpolate(self.bi_search(0, self.rows - 1, T),
                                    T)[self.selected_band + 1]
        else:
            return np.ones(len(self.selected_band)) * 1000

    def r2bt(self, radiance):
        """
		根据波段号(从1开始计数)BandIdx返回黑体温度T
		"""
        radiance = np.array(radiance, dtype=float)
        T = np.empty_like(radiance)
        if self.band_count != T.shape[0]:
            raise Exception("输入数据和选择波段维度信息不匹配!")
        else:
            for idx in range(self.band_count):
                self.selected_col = self.selected_band[idx] + 1
                if self.boundary_check(radiance[idx]):
                    T[idx] = self.interpolate(
                        self.bi_search(0, self.rows - 1, radiance[idx]),
                        radiance[idx])[0]
                else:
                    T[idx] = -999
        return T

    def boundary_check(self, val):
        if self.data[0][self.selected_col] > val or self.data[self.rows - 1][
                self.selected_col] < val:
            # LOG.info("data value out of the range of lut!")
            return False
        return True

    def interpolate(self, idx, val):
        lt, rt = idx
        if lt == rt:
            return self.data[idx[0], :]
        else:
            fa = self.data[idx[0], :].reshape(self.cols)
            fb = self.data[idx[1], :].reshape(self.cols)
            a = self.data[idx[0], self.selected_col]
            b = self.data[idx[1], self.selected_col]
            return (b * fa - a * fb + val * (fb - fa)) / (b - a)

    def __getstate__(self):
        return self.lut, self.selected_bands

    def __setstate__(self, lut, selected_bands):
        self.__init__(lut, selected_bands)


class LookupTable_old(object):
    """Look-up Table implementation
	"""
    def __init__(self, lut):
        self.lut = lut
        self.rows, self.cols = lut.shape
        self.current_col = 0
        self.selected_band_count = self.cols - 1

    def __getstate__(self):
        return self.lut

    def __setstate__(self, lut):
        self.__init__(lut)

    def bt2r(self, TB):
        if np.isnan(TB).any():
            return np.zeros(self.cols - 1)
        self.current_col = 0
        if isinstance(TB, Iterable):
            assert (len(TB) == self.cols - 1)
            r = []
            for i, t in enumerate(TB):
                if self._boundary_check(t):
                    r.append(self._interp_row_col(t, i + 1))
                else:
                    r.append(0)
            return np.asarray(r)
        else:
            if self._boundary_check(TB):
                return self._interp_row(TB)[1:]
            else:
                return np.zeros(self.cols - 1)

    def r2bt(self, Rads):
        if np.isnan(Rads).any():
            return np.zeros(self.cols - 1)
        LOG.debug(
            f'LookupTable.r2bt: input_array dimension check:{len(Rads) == self.cols - 1}'
        )
        TBs = []
        self.current_col = 0
        for rad in Rads:
            self.current_col += 1
            if self._boundary_check(rad):
                TBs.append(self._interp_row(rad)[0])
            else:
                TBs.append(0)
        return np.asarray(TBs)

    def _interp_row_col(self, val, col):
        rows_data = self._bi_search(0, self.rows - 1, val)
        col_x = rows_data[:, self.current_col]
        return float(interp1d(col_x, np.transpose(rows_data)[col])(val))

    def _interp_row(self, val):
        rows_data = self._bi_search(0, self.rows - 1, val)
        col_x = rows_data[:, self.current_col]
        return [
            float(interp1d(col_x, col_data)(val))
            for col_data in np.transpose(rows_data)
        ]

    def _boundary_check(self, val):
        if self.lut[0][self.current_col] > val or self.lut[self.rows - 1][
                self.current_col] < val:
            LOG.debug("data value out of the range of lut!")
            return False
        return True

    def _bi_search(self, l, r, val):
        ''' assuming lut is ranked following the ascending order
		'''
        if r - l == 1:
            LOG.debug('LookupTable bi_search %s, %s', r, l)
            return self.lut[l:r + 1]
        mid = int((r + l) / 2)
        if self.lut[mid][self.current_col] > val:
            return self._bi_search(l, mid, val)
        elif self.lut[mid][self.current_col] < val:
            return self._bi_search(mid, r, val)
        else:
            return np.array((self.lut[mid], self.lut[mid]))


if __name__ == '__main__':
    pass
