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

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pkulast.config import EMIS_LIB_70, ALG_CFG, SW_ALG, SC_ALG
from pkulast.utils.algorithm import parse_formula
from pkulast.atmosphere.profile import ClearSkyTIGR946
from pkulast.surface.emissivity import Emissivity, UniRel, MultiRel
from pkulast.rtm.model import ModtranReader, ModtranTIGR946, RTMSimul, RTMTrans, ParallelModtranWrapper
from pkulast.exceptions import *

def emiss_variance(f, emiss_lib):
    """ calculate emissivity variance.
    """
    pass


def tes_coefficient(f, relationship='M', emis_lib=EMIS_LIB_70):
    """ calculate split-window coefficients
    """
    rel = None
    if relationship == 'M':
        rel = MultiRel(f, rel=MultiRel.GBR, emis_lib=emis_lib)
    else:
        rel = UniRel(f, emis_lib=emis_lib)
    rel.fit()
    # rel.plot()
    emis_func = rel.emis_func
    lut = f.generate_lut()
    return emis_func, lut


def sc_params(algorithm, bt, **kwargs):
    if algorithm not in SC_ALG:
        raise AlgorithmNotFoundException(f"algorithm {algorithm} not found")
    reqs = ALG_CFG['SC'][algorithm]['requirements']
    for req in reqs:
        if req not in kwargs.keys():
            raise ValueError(f"missing required parameter {req}")
    emis = kwargs['e'] if 'e' in kwargs.keys() else np.ones_like(bt)
    tcwv = kwargs['w'] if 'w' in kwargs.keys() else np.zeros(len(bt))
    mu = kwargs['mu'] if 'mu' in kwargs.keys() else np.ones(len(bt))
    kwargs.update({
                "emis":emis.reshape(-1, 1),
                "e": emis.reshape(-1, 1),
                "T0": 273.15,
                "T": bt.reshape(-1, 1),
                "w": tcwv.reshape(-1, 1),
                "mu": mu.reshape(-1, 1),
            })
    kwargs['constant'] = 0
    pre_expr = ALG_CFG['SC'][algorithm].get("pre_expr")
    if pre_expr:
        for k in pre_expr.keys():
            kwargs[k] = parse_formula(pre_expr[k], kwargs)
    kwargs['constant'] = np.squeeze(kwargs['constant'])
    params = {}
    for k, forms in ALG_CFG['SC'][algorithm].get('optimized_target').items():
        sc_params = []
        for f_str in forms:
            sc_params.append(parse_formula(f_str, kwargs))
        params[k] = np.concatenate(
            sc_params,
            axis=1,
        )
    post_expr = ALG_CFG['SC'][algorithm].get("post_expr")
    if post_expr:
        for k in post_expr.keys():
            kwargs[k] = parse_formula(post_expr[k], kwargs)
    return params, kwargs

def sw_params(algorithm, bt, **kwargs):
    if algorithm not in SW_ALG:
        raise AlgorithmNotFoundException(f"algorithm {algorithm} not found")
    reqs = ALG_CFG['SW'][algorithm]['requirements']
    for req in reqs:
        if req not in kwargs.keys():
            raise ValueError(f"missing required parameter {req}")
    emis = kwargs['e'] if 'e' in kwargs.keys() else np.ones_like(bt)
    tcwv = kwargs['w'] if 'w' in kwargs.keys() else np.zeros(len(bt))
    mu = kwargs['mu'] if 'mu' in kwargs.keys() else np.ones(len(bt))
    kwargs.update({
                "emis": emis.reshape(-1, 1),
                "e": ((emis[:, 0] + emis[:, 1]) / 2.0).reshape(-1, 1),
                "de": ((emis[:, 0] - emis[:, 1])).reshape(-1, 1),
                "e11": emis[:, 0],
                "e12": emis[:, 1],
                "T0": 273.15,
                "T11": bt[:, 0],
                "T12": bt[:, 1],
                "dT": ((bt[:, 0] - bt[:, 1])).reshape(-1, 1),
                "w": tcwv.reshape(-1, 1),
                "mu": mu.reshape(-1, 1),
            })
    kwargs['constant'] = 0
    pre_expr = ALG_CFG['SW'][algorithm].get("pre_expr")
    if pre_expr:
        for k in pre_expr.keys():
            kwargs[k] = parse_formula(pre_expr[k], kwargs)
    kwargs['constant'] = np.squeeze(kwargs['constant'])
    params = {}
    for k, forms in ALG_CFG['SW'][algorithm].get('optimized_target').items():
        sw_params = []
        for f_str in forms:
            sw_params.append(parse_formula(f_str, kwargs))
        params[k] = np.concatenate(
            sw_params,
            axis=1,
        )
    post_expr = ALG_CFG['SW'][algorithm].get("post_expr")
    if post_expr:
        for k in post_expr.keys():
            kwargs[k] = parse_formula(post_expr[k], kwargs)
    return params, kwargs

def sc_coefficient(rsr,
                   alg='RTE',
                   dTs=range(-10, 25, 5),
                   emis_lib=EMIS_LIB_70,
                   profile_lib="TIGR946",
                   cwv_intervals=[[0.0, 2.5], [2.0, 3.5], [3.0, 4.5], [4.0, 5.5], [5.0, 6.3], [0, 6.3]],
                   mu=[
                       1,
                   ],
                   dataset=None,
                   store_dataset=False,):
    """ calculate single-channel coefficients
    """
    if alg not in SC_ALG:
        raise AlgorithmNotFoundException(f"algorithm {alg} not found")
    assert rsr.selected_band_count == 1 # "only support single-channel"
    alg = ALG_CFG['SC'][alg]
    md = RTMSimul(rsr, dTs, emis_lib, profile_lib, mu)
    if dataset is None:
        simulated_dataset = md.simulate(selected_tcwv=[0, 10])
    else:
        simulated_dataset = dataset
    # if 'w' in alg['requirements']:
    #     cwv_intervals = [
    #         [0, 10],
    #     ]
    models = []
    params_ab = rsr.derivation_ab()
    for interval in cwv_intervals:
        indices = md.get_indices(selected_tcwv=interval)
        # indices需要考虑
        kwargs = {
            "e": simulated_dataset["emis"][indices].reshape(-1, 1),
            "T0": 273.15,
            "T": simulated_dataset["bt"][indices].reshape(-1, 1),
            "L": simulated_dataset["rad"][indices].reshape(-1, 1),
            "w": simulated_dataset["w"][indices].reshape(-1, 1),
            "tau":simulated_dataset["tau"][indices].reshape(-1, 1),
            "Lu":simulated_dataset["Lu"][indices].reshape(-1, 1),
            "Ld":simulated_dataset["Ld"][indices].reshape(-1, 1),
            "Ta":simulated_dataset["Ta"][indices].reshape(-1, 1),
            "mu": simulated_dataset["mu"][indices].reshape(-1, 1),
            "a": params_ab[0][0],
            "b": params_ab[0][1],
            "blambda": rsr.b_lambda()[0],
            "ttau":simulated_dataset["tau"][indices].reshape(-1, 1),
            "tLST":simulated_dataset["lst"][indices],
        }
        kwargs['constant'] = 0
        pre_expr = alg.get("pre_expr")
        if pre_expr:
            for k in pre_expr.keys():
                kwargs[k] = parse_formula(pre_expr[k], kwargs) 
        kwargs['constant'] = np.squeeze(kwargs['constant'])
        fit_intercept= alg.get('optimized_params').get('fit_intercept')
        model = LinearRegression(fit_intercept=fit_intercept)
        coefs = []
        for k, forms in alg.get('optimized_target').items():
            sc_params = []
            for f_str in forms:
                sc_params.append(parse_formula(f_str, kwargs))
            params = np.concatenate(
                sc_params,
                axis=1,
            )
            model.fit(params, kwargs[k] - kwargs['constant'])
            predict_key = 'p' + k[1:]
            kwargs[predict_key] = model.predict(params) + kwargs['constant']
            coefs.append(np.append(model.intercept_, model.coef_))
        post_expr = alg.get("post_expr")
        if post_expr:
            for k in post_expr.keys():
                kwargs[k] = parse_formula(post_expr[k], kwargs)
        tLST = kwargs['tLST']
        pLST = np.squeeze(parse_formula(alg.get('formula'), kwargs))
        models.append({
            "label": "{}_{}-{}".format(rsr.sensor, *interval),
            "coeffs": coefs,
            "residuals": pLST - tLST,
            "pLST": pLST,
            "tLST": tLST,
            "RMSE": np.sqrt(mean_squared_error(tLST, pLST)),
            "R2": r2_score(tLST, pLST),
            "dataset": simulated_dataset if store_dataset else None,
            "params": kwargs,
        })
    return models

cwv_intervals = [[0.0, 2.5], [2.0, 3.5], [3.0, 4.5], [4.0, 5.5], [5.0, 6.3], [0, 6.3]]

def sw_coefficient(rsr,
                   alg,
                   dTs=range(-10, 25, 5),
                   emis_lib=EMIS_LIB_70,
                   profile_lib="TIGR946",
                   cwv_intervals=[[0.0, 2.5], [2.0, 3.5], [3.0, 4.5], [4.0, 5.5], [5.0, 6.3], [0, 6.3]],
                   mu=[1, ],
                   dataset=None,
                   store_dataset=False,):
    """ calculate split-window coefficients.
    """
    if alg not in SW_ALG:
        raise AlgorithmNotFoundException(f"algorithm {alg} not found")
    assert rsr.selected_band_count == 2 # "only support split-window"
    alg = ALG_CFG['SW'][alg]
    md = RTMSimul(rsr, dTs, emis_lib, profile_lib, mu)
    if dataset is None:
        simulated_dataset = md.simulate(
            selected_tcwv=[0, 10])
    else:
        simulated_dataset = dataset
    # if 'w' in alg['requirements']:
    #     cwv_intervals = [
    #         [0, 10],
    #     ]
    params_ab = rsr.derivation_ab()
    models = []
    for interval in cwv_intervals:
        indices = md.get_indices(selected_tcwv=interval)
        bt, emis, = simulated_dataset["bt"][indices], simulated_dataset["emis"][
            indices], 
        kwargs = {
            "e": ((emis[:, 0] + emis[:, 1]) / 2.0).reshape(-1, 1),
            "de": ((emis[:, 0] - emis[:, 1])).reshape(-1, 1),
            "e11": emis[:, 0].reshape(-1, 1),
            "e12": emis[:, 1].reshape(-1, 1),
            "T0": 273.15,
            "T11": bt[:, 0].reshape(-1, 1),
            "T12": bt[:, 1].reshape(-1, 1),
            "dT": ((bt[:, 0] - bt[:, 1])).reshape(-1, 1),
            "w": simulated_dataset["w"][indices].reshape(-1, 1), 
            "ttau11":simulated_dataset["tau"][:, 0][indices].reshape(-1, 1),
            "ttau12":simulated_dataset["tau"][:, 1][indices].reshape(-1, 1),
            "a11": params_ab[0][0],
            "b11": params_ab[0][1],
            "a12": params_ab[1][0],
            "b12": params_ab[1][1],
            "mu": simulated_dataset["mu"][indices].reshape(-1, 1),
            "tLST":simulated_dataset["lst"][indices],
        }
        kwargs['constant'] = 0
        pre_expr = alg.get("pre_expr")
        if pre_expr:
            for k in pre_expr.keys():
                kwargs[k] = parse_formula(pre_expr[k], kwargs)
        kwargs['constant'] = np.squeeze(kwargs['constant'])
        fit_intercept= alg.get('optimized_params').get('fit_intercept')
        model = LinearRegression(fit_intercept=fit_intercept)
        coefs = []
        for k, forms in alg.get('optimized_target').items():
            sw_params = []
            for f_str in forms:
                sw_params.append(parse_formula(f_str, kwargs))
            params = np.concatenate(
                sw_params,
                axis=1,
            )
            model.fit(params, kwargs[k] - kwargs['constant'])
            predict_key = 'p' + k[1:]
            kwargs[predict_key] = model.predict(params) + kwargs['constant']
            coefs.append(np.append(model.intercept_, model.coef_))
        post_expr = alg.get("post_expr")
        if post_expr:
            for k in post_expr.keys():
                kwargs[k] = parse_formula(post_expr[k], kwargs)
        tLST = kwargs['tLST']
        pLST = np.squeeze(parse_formula(alg.get('formula'), kwargs))
        models.append({
            "label": "{}_{}-{}".format(rsr.sensor, *interval),
            "coeffs": coefs,
            "residuals": pLST - tLST,
            "pLST": pLST,
            "tLST": tLST,
            "RMSE": np.sqrt(mean_squared_error(tLST, pLST)),
            "R2": r2_score(tLST, pLST),
            "dataset": simulated_dataset if store_dataset else None,
            "params": kwargs,
        })
    return models

def SWCVR(trans):
    ratio = (trans[:, 0] / trans[:, 1]).reshape(-1, 1)
    params = np.concatenate(
        (ratio, ratio * ratio),
        axis=1,
    )
    return params


def wv_coefficient(rsr):
    """ water vapor coefficients
    wv = a * tau^2  + b * tau + c
    """
    # tigr = ModtranTIGR946()
    # trans = tigr.integral_t(rsr)
    md = RTMTrans(rsr)
    tcwv, trans = md.load()
    params = SWCVR(trans)
    model = LinearRegression()
    model.fit(params, tcwv)
    coefs = np.append(model.intercept_, model.coef_)
    res = model.predict(params) - tcwv
    return coefs, res, model


if __name__ == '__main__':
    from pkulast.remote.sensor import RSR
    rsr = RSR("landsat_8_tirs")
    rsr.plot()