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
Atmospheric correction.
=======================
TODO: add implemnetation details.
Provides atmospheric correction for VNIR SWIR MIR TIR data  .
"""

class AerosolOpticalDepth(object):
    ''' Aerosol Optical Depth(AOD) Retrieval
	1.
	https://aeronet.gsfc.nasa.gov/
	AeroNet CE318 aod@500nm / visibility
	Network from CAS

	'''
    def __init__(self):
        pass


class WaterVapor(object):
    ''' Water Vapor Retrieval

	1.
	tau(lambda_wv) = L(lambda_wv) / L(lambda_nwv)
	or tau(lambda_wv) = L(lambda_wv) / (c1*L(lambda_nwv1) + c2*L(lambda_nwv2))

	then wv = (alpha - ln(tau / beta))^2

	2.
	https://aeronet.gsfc.nasa.gov/
	AeroNet CE318  WV@946nm

	3.
	Split-Window covariance-variance ratio

	tau_j / tau_i = epsilon_i * R_ji / epsilon_j aeq R_ji = convariance / variance
	R = tau_j / tau_i
	wv = aR^2 + bR + c
	MODIS Landsat S3/SLSTR GF5/VIMS HJ-2 0.5g/cm^2

	high calibretion requirements and low noise level
	perform poorly in water body

	4. 
	NWP based method
	'''
    def __init__(self):
        pass
