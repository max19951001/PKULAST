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

""" atmosphere utilities"""

import math
import numpy as np
import xarray as xr
from datetime import datetime
from collections import defaultdict
from subprocess import check_output
from scipy.interpolate import interp1d, griddata
from pkulast.constants import GRAV_ACC, GAS_CONST, MOL_MASS


# Refractor Code

def merge_and_save_nc_files(nc_files, filename):
    final_ds = xr.Dataset()
    for name in nc_files:
        remote_ds = xr.open_dataset(name)
        final_ds = xr.merge([final_ds, remote_ds])
        remote_ds.close()
    encoding = {v: {'zlib': True, 'complevel': 4} for v in final_ds.data_vars}
    final_ds.to_netcdf(filename, encoding=encoding)
    final_ds.close()


def merge_grib_files(grib_files, filter_by_keys=None, variables=None):
    final_ds = None
    for name in grib_files:
        if filter_by_keys:
            remote_ds = xr.open_dataset(name, engine='cfgrib',filter_by_keys=filter_by_keys) #, backend_kwargs={'errors': 'ignore'}
        else:
            remote_ds = xr.open_dataset(name, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        if variables:
            remote_ds = remote_ds[variables]
        if final_ds:
            final_ds = xr.merge([final_ds, remote_ds], compat='minimal')
        else:
            final_ds = remote_ds
        remote_ds.close()
    return final_ds


def pressure_to_altitude(pressures, T0, p0):
    ''' Convert pressures to altitude
	0 - 84.8520 geopotential height
	'''
    is_reversed = np.array_equal(sorted(pressures), pressures)
    if is_reversed:
        pressures = list(reversed(pressures))
    ret = HPConverter(T0, p0).pressure_to_altitude(pressures)
    return list(reversed(ret)) if is_reversed else ret

def rh2sh(pressures, RH, TMP):
    """ convert relative humidity to specific humidity
	"""
    Q = []
    for i, p in enumerate(pressures):
        if RH[i] < 1e-15:
            q = 0
        else:
            es = 6.112 * math.exp(17.67 * (TMP[i] - 273.15) / (243.5 + (TMP[i] - 273.15)))
            q = RH[i] * (0.62197 * es / (p - 0.378 * es)) / 100
        # print(q)
        Q.append(q)
    # print(Q)
    return Q

def sh2rh(pressures, SH, TMP):
    """ convert specific humidity to relative humidity
	"""
    R = []
    for i, p in enumerate(pressures):
        es = 6.112 * math.exp(17.67 * (TMP[i] - 273.15) / (243.5 + (TMP[i] - 273.15)))
        e = SH[i] * p / (0.378 * SH[i] + 0.622)
        rh = e / es
        rh = 1 if rh > 1 else rh
        rh = 0 if rh < 0 else rh
        R.append(rh)
    return R

def get_tcwv(x, p):
    '''
	x: g/g mixing ratio
	p: pressure hPa
	return: kg / m^2 == mm == 10 g/cm^2
	'''
    return abs(np.trapz(x, p) * 10 / 9.80665)

# def atmcm2

def vibeta(P, X, PSFC=1100, PBOT=1100, PTOP=0, LINLOG=1, XMSG=-1):
    ZERO = PZERO = XZERO = 0
    is_reversed = np.array_equal(sorted(P), P)
    if is_reversed:
        P = list(reversed(P))
        X = list(reversed(X))
    KLVL = 300
    IER = 0
    NLEV = len(P)
    if NLEV < 3:
        IER += 1
    if KLVL < 2 * NLEV:
        IER += 10
    if PSFC == XMSG:
        IER += 100
    if PTOP > PBOT:
        IER = IER + 10000
    if IER != 0:
        VINT = XMSG
        return
    NLL = 0
    NLX = 0
    PFLAG = False
    PP = np.full(KLVL, XMSG, dtype=float)
    XX = np.full(KLVL, 0, dtype=float)

    # 去除X P中 nodata点并且进行积分下限筛选，生成PP XX
    for NL in range(1, NLEV + 1):
        if P[NL - 1] != XMSG and X[NL - 1] != XMSG and P[NL - 1] < PSFC:
            NLX = NL
            NLL = NLL + 1
            PP[NLL] = P[NL - 1]
            XX[NLL] = X[NL - 1]
            if PP[NLL] == PTOP:
                PFLAG = True  # P中有效值达到了积分上限
    NLMAX = NLL  # P中有效值的个数

    # 总有效值的个数必须大于等于3
    if NLMAX < 3:
        IER = -999
        VINT = XMSG
        raise RuntimeError('总有效值的个数必须大于等于3')
        return None

    # 如果有效值数目小于总数据个数，将P和X最后一个有效值之后的无效值填充到PP和XXNLEV
    if NLMAX < NLEV:
        for NL in range(NLX, NLEV):
            PP[NLL + 1] = P[NL]
            XX[NLL + 1] = 0
            NLL = NLL + 1

    if PP[NLMAX] > PTOP:  #说明有效值最小值大于积分上限，需要进行插值处理
        NLMAX += 1
        PP[NLMAX] = PTOP  # 强行加入PTOP层 使之达到积分上限，并插值赋值
        # assume p=0, x=0 is next level above.
        # 假设积分上限上一层 p=0, x=0
        if LINLOG == 1:
            SLOPE = (XX[NLMAX - 1] - XZERO) / (PP[NLMAX - 1] - PZERO)
            XX[NLMAX] = XZERO + (PP[NLMAX] - PZERO) * SLOPE
        elif LINLOG == 2:
            SLOPE = (XX[NLMAX - 1]) / math.log(PP[NLMAX - 1])
            XX[NLMAX] = XZERO + math.log(PP[NLMAX] - PZERO) * SLOPE
    elif not PFLAG:  # 说明积分上限在PP有效数据范围里，但是该层并不在离散层中，需要插入该层
        BETA = np.array(PP)
        DELP = np.array(XX)
        # 找到PTOP应该插入的位置
        for NL in range(1, NLMAX - 1):
            if PP[NL] > PTOP and PP[NL + 1] < PTOP:
                NLTOP = NL + 1
                NLSAV = NLTOP
                break

        # 插入该层
        NL1 = NLTOP
        NL2 = NLTOP - 1
        if LINLOG == 1:
            SLOPE = (XX[NL2] - XX[NL1]) / (PP[NL2] - PP[NL1])
            XX[NLTOP] = XX[NL1] + (PTOP - PP[NL1]) * SLOPE
        elif LINLOG == 2:
            PA = math.log(PP[NL2])
            PC = math.log(PP[NL1])
            SLOPE = (XX[NL2] - XX[NL1]) / (PA - PC)
            XX[NLTOP] = XX[NL1] + (math.log(PTOP) - PC) * SLOPE
        PP[NLTOP] = PTOP

        for NL in range(NLSAV, NLMAX):
            XX[NL + 1] = DELP[NL]
            PP[NL + 1] = BETA[NL]
        NLMAX = NLMAX + 1  # 有效层数+1
    PI = np.full(KLVL + 1, 0, dtype=float)
    XI = np.full(KLVL + 1, 0, dtype=float)
    PI[0] = PBOT

    # 对PI进行奇偶赋值
    for NL in range(1, NLMAX):
        PI[2 * NL - 1] = PP[NL]
        PI[2 * NL] = (PP[NL] + PP[NL + 1]) * 0.5
    PI[2 * NLMAX - 1] = PP[NLMAX]
    PI[2 * NLMAX] = (PP[NLMAX] + ZERO) * 0.5
    NLTOP = 0
    for NL in range(1, 2 * NLMAX):
        if PI[NL] == PTOP:
            NLTOP = NL
            break

    #找到PTOP对应的index NLTOP
    if NLTOP == 0 or NLTOP % 2 == 0:
        raise RuntimeError('P结构不正确')
        return None
    # 对PTOP进行赋值
    PI[NLTOP] = PTOP
    PI[NLTOP + 1] = (PTOP + ZERO) * 0.5

    if PSFC != XMSG:
        PSFCX = PSFC
    else:
        PSFCX = PP[1]
    # 对XI进行赋值处理
    XI = DINT2P2(PP, XX, NLMAX, PI, XI, NLTOP, LINLOG, XMSG)

    DELP = np.full(KLVL + 1, 0, dtype=float)
    BETA = np.full(KLVL + 1, 0, dtype=float)

    for NL in range(1, NLTOP + 1):
        DELP[NL] = PI[NL - 1] - PI[NL + 1]

    for NL in range(1, NLTOP + 1):
        if PI[NL - 1] > PSFCX and PI[NL + 1] < PSFCX:
            BETA[NL] = (PSFCX - PI[NL + 1]) / (PI[NL - 1] - PI[NL + 1])
        elif PI[NL - 1] > PTOP and PI[NL + 1] < PTOP:
            BETA[NL] = (PI[NL - 1] - PTOP) / (PI[NL - 1] - PI[NL + 1])
        elif PI[NL] < PBOT and PI[NL] > PTOP:
            BETA[NL] = 1

    VINT = 0.0
    for NL in range(1, NLTOP + 1, 2):
        VINT = VINT + BETA[NL] * XI[NL] * DELP[NL]
    return VINT


def DINT2P2(PIN, XIN, NPIN, POUT, XOUT, NPOUT, IFLAG, XMSG):
    '''
    c .   pin    - input pressure levels. The pin should
    c .            be in decending order (e.g., 1000,900,825,..)
    c .            pin(1)>pin(2)>...>pin(npin)
    c .   xin    - data at input pressure levels
    c .   npin   - number of input pressure levels
    c .   pout   - output pressure levels (input by user)
    c .            decending order required.
    c .   xout   - data at output pressure levels
    c .   npout  - number of output pressure levels
    c .   iflag  - if iflag=1 user linear interp in pressure
    c .            if iflag=2 user linear interp in ln(pressure)
    c .   xmsg   - missing data code. if none, set to some number
    c .            which will not be encountered (e.g., 1.e+36)
    c .   ier    - error code
    '''
    NPLVL = 200
    P = np.zeros(NPLVL)
    X = np.zeros(NPLVL)
    SLOPE = PA = PB = PC = 0
    IER = 0
    if NPIN < 1 or NPOUT < 1:
        IER += 1
    if IFLAG < 1 or IFLAG > 2:
        IER += 10
    if NPIN > NPLVL:
        IER += 100

    if IER != 0:
        for NP in range(NPOUT):
            XOUT[NP] = XMSG
        return False

    NL = 0
    for NIN in range(1, NPIN + 1):
        if XIN[NIN] != XMSG and PIN[NIN] != XMSG:
            NL = NL + 1
            P[NL] = PIN[NIN]
            X[NL] = XIN[NIN]
    NLMAX = NL
    if NLMAX == 0:
        IER += 1000
        return

    for NP in range(1, NPOUT + 1):
        XOUT[NP] = XMSG
        for NL in range(1, NLMAX + 1):
            if POUT[NP] == P[NL]:
                XOUT[NP] = X[NL]
            elif NL < NLMAX:
                if POUT[NP] < P[NL] and POUT[NP] > P[NL + 1]:
                    if IFLAG == 1:
                        SLOPE = (X[NL] - X[NL + 1]) / (P[NL] - P[NL + 1])
                        XOUT[NP] = X[NL + 1] + SLOPE * (POUT[NP] - P[NL + 1])
                    else:
                        PA = math.log(P[NL])
                        PB = math.log(POUT[NP])
                        PC = math.log(P[NL + 1])
                        SLOPE = (X[NL] - X[NL + 1]) / (PA - PC)
                        XOUT[NP] = X[NL + 1] + SLOPE * (PB - PC)
    return XOUT


def get_tcwv_beta(x, p, pbot = 1100, ptop=0):
    """ vectical integration using beta methods
	vint = vibeta(p, x, linlog, psfc, pbot, ptop) g cm^2
	"""
    # pbot = 1100 # bottom pressure
    # ptop = 0 # top pressure
    # ensure pressure following descending order(from bottom to toa)
    is_reversed = np.array_equal(sorted(p), p)
    if is_reversed:
        p = list(reversed(p))
        x = list(reversed(x))
    psfc = p[0] # surface pressure
    plst = p[-1] # last level pressure
    nlev = len(x)
    pi = np.zeros(2 * nlev + 1)
    xi = np.zeros(2 * nlev)
    delp = np.zeros(2 * nlev)
    beta = np.zeros(2 * nlev)

    # set pi and xi
    pi[0] = pbot
    for i in range(nlev):
        xi[2 * i + 1] = x[i] / (x[i] + 1)
        pi[2 * i + 1] = p[i]
        if i >= 1:
            pi[2 * i] = (p[i - 1] + p[i]) * 0.5
    pi[2 * nlev] = (0 + p[nlev - 1]) * 0.5

    # set delp and beta
    for i in range(1, 2 * nlev, 2):
        delp[i] = pi[i-1] - pi[i+1]
        if pi[i-1] > psfc and pi[i+1] < psfc:
            beta[i] = (psfc - pi[i+1]) / delp[i]
        elif pi[i-1] > plst and pi[i+1] < plst:
            beta[i] = (pi[i-1] - plst) / delp[i]
        else:
            beta[i] = 1
    vint = 0
    for i in range(1, 2 * nlev, 2):
        vint += beta[i] * xi[i] * delp[i]
    return vint * 10 / 9.80665

def get_g0(lat):
    """ get gravitational acceleration
	"""
    phi = math.radians(lat) # latitude(deg)
    g0 = 9.80620 * (1 - 0.0026642 * math.cos(2 * phi) + 0.0000058 * math.cos(2 * phi) **2)
    g0 = 9.80665 * (1 - 0.00265 * math.cos(2 * phi) )
    g0 = 9.780318 * (1 + 0.0053024 * math.sin(2 * phi) - 0.0000059 * math.sin(phi) ** 2)  #m/s^2
    g0 = 9.78046 * (1 + 0.0052884 * math.sin(2 * phi) - 0.0000059 * math.sin(phi) ** 2)  #m/s^2
    return g0

def h2z(h):
    r0 = 6356766
    return r0 * h /(r0 - h) / 1000

def z2h(z):
    r0 = 6356766
    return r0 * z / (r0 + z) / 1000

def get_height(P, TMP, RH):
    '''
	ΔH= Rd/G Tv * (lnp1-lnp2)
	Tv = T^hat(1 + 0.00378 U^hat * E^hat / p^hat)

	U^hat: 平均相对湿度(%)
	E^hat: 温度T^hat时的饱和蒸汽压(hPa)
	p^hat: 平均气压
	层间平均绝对温度T^hat(K)与层间平均摄氏温度t^hat(℃)
	马格努斯饱和水汽压公式
	E^hat = 6.112 exp(17.62 t^hat / (243.12 + t^hat))
	p^hat = exp((lnp1 + lnp2)/2) = sqrt(q1*q2)

	g0 = 9.80665
	Rd = 287.05287 # 干空气气体常数
	Rd / g0 = 29.27096

	g0_phi = 9.80620(1-0.0026642cos(2phi) + 0.0000058cos^2(2phi))


	g_phi_h = 9.80665(1-0.00265cos(2phi)) / (1 + 2h/R)

	g_phi = 9.80620(1-0.0026642cos(2phi) + 0.0000058cos^2(2phi))/ (1 + 2h/R)

	g = g0_phi / (1 + 2h / R)

	g = g0_phi * (R_phi / R_phi+h)^2

	R_phi = 2 * g0_phi / (3.085462*1e-6 + 2.27* 1e-9 cos(2phi))

	ΔH= Rd/G Tv * (lnp1-lnp2) = 287.05287/G Tv * (lnp1-lnp2)
	'''
    is_reversed = np.array_equal(sorted(P), P)
    if is_reversed:
        P = list(reversed(P))
        TMP = list(reversed(TMP))
        RH = list(reversed(RH))
    g0 = 9.80665
    Rd = 287.05287
    H = [2, ]
    nlev = len(P)
    for i in range(nlev-1):
        T_hat = (TMP[i] + TMP[i+1]) / 2
        # reference https://wenku.baidu.com/view/0d5cc624657d27284b73f242336c1eb91a3733b5.html?re=view
        E_hat = 6.10695 * math.pow(10, 7.5927 * (TMP[i] - 273.15) / (240.72709 + (TMP[i] - 273.15)))
        # E_hat = 6.112 * math.exp(17.62 * (TMP[i] - 273.15) / (243.12 + (TMP[i] - 273.15)))
        # es=610.94.*exp(17.625.*Tc./(Tc+243.04));
        U_hat = (RH[i] + RH[i + 1]) / 2 #/ 100
        p_hat = math.sqrt(P[i] * P[i+1])
        Tv = T_hat * (1 + 0.00378 * U_hat * E_hat / p_hat)
        delH = Rd / g0 * Tv * math.log(P[i] / P[i+1])
        last_H = H[-1]
        H.append(last_H + delH)
    Z = list(map(h2z, H))
    return list(reversed(Z)) if is_reversed else Z

def mmr2rh(h2o, TMP, P):
    """ Converting Mass Mixing Ratio(g/g) to Relative Humidity(%)
	-45 - 60 deg
	"""
    RH = []
    for i, h in enumerate(h2o):
        E = 6.10695 * math.pow(10, 7.5927 * (TMP[i] - 273.15) / (240.72709 + (TMP[i] - 273.15)))
        # E = 6.112 * math.exp(17.62 * (TMP[i] - 273.15) / (243.12 + (TMP[i] - 273.15)))
        e = h * P[i] / (0.622 + h)
        RH.append(e * 100 / E)
    return RH


from pkulast.utils.physics.wxparams import RH_to_Td, Absolute_Humidity
def rh2ah(RH, TMP, P):
    AH = []
    for i, rh in enumerate(RH):
        es = 6.112 * math.exp(17.62 * (TMP[i] - 273.15) / (243.12 +
                                                            (TMP[i] - 273.15)))
        # rvs = 0.622 * es / (P[i] - es)
        # rv = rh / 100. * rvs
        # qv = rv / (1 + rv)
        e = es * rh / 100
        ah = 2.16674 * e * 100 / TMP[i]
        AH.append(ah)



# def ah2rh(AH, TMP, P):
#     RH = []
#     for i, rh in enumerate(AH):
#         es = 6.112 * math.exp(17.62 * (TMP[i] - 273.15) / (243.12 +
#                                                            (TMP[i] - 273.15)))
#         # rvs = 0.622 * es / (P[i] - es)
#         # rv = rh / 100. * rvs
#         # qv = rv / (1 + rv)
#         e = es * rh / 100
#         ah = 2.16674 * e * 100 / TMP[i]
#         AH.append(ah)

# es = 611.2*exp(17.67*(T-273.15)/(T-29.65))
# rvs = 0.622*es/(p - es)
# rv = RH/100. * rvs
# qv = rv/(1 + rv)
# AH = qv*rho

def str2fp(num):
    try:
        return float(num.strip())
    except Exception as ex:
        raise ValueError(f"{ex}, input string can not converted to float point number")

def str2int(num):
    try:
        return int(num.strip())
    except Exception as ex:
        raise ValueError(f"{ex},input string can not converted to integer")

def str2date(date_str):
    year = 1900 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:])
    if day:
        return datetime(year, month, day)
    else:  # dd is not defined
        return datetime(year, month, day + 1)
        # return f"{year}-{month:02d}-na"

def spatial_interpolation(lat_lons, values, site):
    ''' Spatial interpolation from values associated with four points.
	'''
    return float(griddata(lat_lons, values, [site], method='linear'))

def temporal_interpolation(times, vals, time):
    ''' Temporal interpolation from values associated with two times.
	'''
    return float(interp1d(times, vals)(time))

def run_command(cmd, shell=True):
    ''' Run command and check output.
	'''
    return check_output(cmd, shell=shell).decode('utf-8').strip().split('\n')

def fix_null(_dict, P, H):
    '''remove null value
	'''
    result = defaultdict()
    layer_number = len(P)
    for lyr in range(layer_number):
        p = P[lyr]
        h = H[lyr]
        if p in _dict:
            result[p] = _dict[p]
        else:
            lt = rt = lyr
            while P[lt] not in _dict and lt > 0:
                lt -= 1
            while P[rt] not in _dict and rt < layer_number - 1:
                rt += 1
            if lt == rt:
                result[p] = 0
            else:
                V1 = 1e-16
                V2 = 1e-16
                if P[lt] in _dict:
                    V1 = _dict[P[lt]]
                if P[rt] in _dict:
                    V2 = _dict[P[rt]]
                result[p] = V1 + (V2 - V1) * (h - H[lt]) / (H[rt] - H[lt])
    return result

def atmosphere_model(lat, month):
    '''
	根据Acquire Time确定MODTRAN中大气类型(Atmosphere Model)对应字符串
	'''
    AM_Tropical = "AAH1C111111111 1 "       # 热带
    AM_MidLatSummer = "AAH2C222222222 2 "   # 中纬度冬季
    AM_MidLatWinter = "AAH3C333333333 3 "   # 中纬度夏季
    AM_SubArcticSummer = "AAH4C444444444 4 "# 极地夏季
    AM_SubArcticWinter = "AAH5C555555555 5 "# 极地冬季
    if lat <= 25:
        return AM_Tropical
    elif (lat <= 65):
        if  month >= 4 and month < 10:
            return AM_MidLatSummer
        else:
            return AM_MidLatWinter
    else:
        if month >= 4 and month < 10:
            return AM_SubArcticSummer
        else:
            return AM_SubArcticWinter

class HPConverter(object):
    """ Convert geometric height and air pressure
	reference: https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf
	
	# km geopotential height
	Hb_0_6 = [0, 11, 20, 32, 47, 51, 71, 84.8520]
	Lb_0_6 = [-6.5, 0, 1.0, 2.8, 0, -2.8, -2.0]

	# km geometric height
	Zb_7_12 = [86, 91, 110, 120, 500, 1000]
	Lb_8_11 = [0, None, 12.0, None, None]

	'''
	根据压强(Pressures Array)和下垫面温度(T0)计算压强对应海拔高度
	'''
	# 在计算探空气球的漂移信息时：
	# 位势高度就是用与大气中某点的重力位势成正比的位势米来表示的该点高度.用位势米表示的位势高度等于用（几何〕米表示的几何高度的g/9.8倍,其中g为当地重力加速度.
	# 把地球当作旋转椭球,公式为：
	# g=9.7803(1+0.0053024sin^2ψ-0.000005sin^2ψ) m/s^2; （1979修订公式）,
	# 式中ψ为物体所在处的地理纬度.例如,在赤道=0,g=9.78m/s^2;,在两极=90°,g=9.83m/s^2;.
	# 在计算探空气球漂移信息时，不考虑曲率的影响，也不考虑重力加速度随高度和纬度的变化。

	#Pressures =[1000.0,925,850,700,600,500,400,300,250,200,150,100,70,50,30,20,10]

	# Pressures =[1013, 902.103, 802, 710, 628, 554.0, 487.0, 426.0, 372.0, 324.0, 281.0, 243.0,$
	#             209.0, 179.0, 153.0, 130.0, 111.0, 95.0,81.2, 69.5, 59.5, 51.0, 43.7, 37.6,$
	#              32.2, 27.7, 20.45, 13.2, 6.52, 3.33, 0.951 ,  0.067]

	# https://www.mide.com/air-pressure-at-altitude-calculator
	# for altitude less than 11km
	#       h = hb + Tb/Lb * [(p/pb)^(-R*Lb/(g0*M)) -1]
	# for altitude larger than 11km
	#       h = hb + R * Tb * ln(P/P0) / (-go * M)
	# pb = 1013.24       # static pressure (pressure at sea level) [hPa]
	# Tb =  273.16+15    #standard temperature (temperature at sea level) [K]
	# Tb = T0  #           #standard temperature (temperature at sea level) [K]
	Lb = -0.0065         # standard temperature lapse rate [K/m] = -0.0065 [K/m]
	# h is height above sea level

	hb = 0.0  # height at the bottom of atmospheric layer [m]
	R = 8.31432  # universal gas constant = 8.31432 [N*m/(mol.K)]
	g0 = 9.80665  # gravitational acceleration constant,m/s2
	M = 0.0289644  # molar mass of Earth's air = 0.0289644 [kg/mol]

	"""
    def __init__(self, T0, p0):
        ''' Parmas need to be changed Tb, pb, Lb
		'''
        self.Hb_lst = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852] # height at the bottom of atmospheric layer [m]
        self.Lb_lst = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0028, -0.0020] # temperature lapse rate [K/m]
        self.Tb_lst = []
        self.Pb_lst = []
        self.Tb_lst.append(T0)
        self.Pb_lst.append(p0)
        self._load()

    def _load(self):

        # Tb
        for i, Lb in enumerate(self.Lb_lst):
            Tb = self.Tb_lst[-1]
            self.Tb_lst.append(Tb + Lb*(self.Hb_lst[i+1] - self.Hb_lst[i]))
        # Pb
        for i, Lb in enumerate(self.Lb_lst):
            Pb = self.Pb_lst[-1]
            self.Pb_lst.append(self._get_bottom_pressure(self.Hb_lst[i+1], self.Hb_lst[i], self.Tb_lst[i], Pb, self.Lb_lst[i]))

    def pressure_to_altitude(self, pressures):
        geometric_height = []
        self.layer_index = 0
        for p in pressures:
            geometric_height.append(self._get_height(p) / 1000.0)
        return geometric_height

    def _get_bottom_pressure(self, H, hb, Tb, pb, Lb):
        if Lb == 0:
            return pb * math.exp((-GRAV_ACC * MOL_MASS * (H-hb))/(GAS_CONST * Tb))
        return pb * (1 + Lb * (H - hb) / Tb)**(-GRAV_ACC * MOL_MASS /(GAS_CONST * Lb))

    def _get_height(self, p):
        if self.layer_index >= 6:
            return 100000
        upper_height = self.Hb_lst[self.layer_index + 1]
        hb = self.Hb_lst[self.layer_index]
        Tb = self.Tb_lst[self.layer_index]
        pb = self.Pb_lst[self.layer_index]
        Lb = self.Lb_lst[self.layer_index]
        if Lb == 0:
            H = hb + GAS_CONST * Tb * math.log(p / pb) / (-GRAV_ACC * MOL_MASS)
        else:
            H = hb + Tb / Lb * ((p / pb)**(-GAS_CONST * Lb / (GRAV_ACC * MOL_MASS)) - 1)
        if H <= upper_height:
            return self._H2Z(H)
        else:
            self.layer_index += 1
            return self._get_height(p)

    def _H2Z(self, H):
        """ convert geopotential altitude to geometric height
		"""
        r0 = 6356766
        return r0 * H /(r0 - H)


if __name__ == '__main__':
    pressures = [1000.0,925,850,700,600,500,400,300,250,200,150,100,70,50,30,20,10, 5, 1, 0.001]
    print(pressure_to_altitude(pressures, 273, 1014.61))
