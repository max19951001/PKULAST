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


""" Constants"""
# Planck radiation equation.
H_PLANCK = 6.62606957 * 1e-34  # SI-unit = [J*s]
K_BOLTZMANN = 1.3806488 * 1e-23  # SI-unit = [J/K]
C_SPEED = 2.99792458 * 1e8  # SI-unit = [m/s]
SIGMA = 5.6697 * pow(10.0,-8) #

# Float precision
EPSILON = 1e-8

# Physics
EARTH_RADIUS = 6356766 # earth radius [m]
GAS_CONST = 8.31432  # universal gas constant = 8.31432 [N*m/(mol.K)]
GRAV_ACC = 9.80665  # gravitational acceleration constant,m/s2
MOL_MASS = 0.0289644  # molar mass of Earth's air = 0.0289644 [kg/mol]
H2O_MASS = 0.01801528  # molar mass of Earth's air = 0.01801528 [kg/mol]
O3_MASS = 0.0479982
N2O_MASS = 0.0440124
CO_MASS = 0.02801
CH4_MASS = 0.01604
O2_MASS = 0.0159994

# Spectrum
WAVE_LENGTH = 'wavelength'
WAVE_NUMBER = 'wavenumber'

ZERO_TEMPERATURE = -273.15    # absolute zero (K)
# EPSILON = sys.float_info.epsilon	# epsilon
MAX_EARTH_TEMPERATURE = 273.15 + 56.7 # max temperature of earth surface(K)




















# -*- coding: utf-8 -*-
"""Collection of physical constants and conversion factors.

Physical constants
==================

=============================== ==============================================
``g``                           Earth standard gravity in :math:`\sf ms^{-1}`
``h``                           Planck constant in :math:`\sf Js`
``k``                           Boltzmann constant in :math:`\sf JK^{-1}`
``c``                           Speed of light in :math:`\sf ms^{-1}`
``N_A``                         Avogadro constant in :math:`\sf mol^{-1}`
``K``, ``zero_celsius``         Kelvin at 0 Celsius
``triple_point_water``          Triple point temperature of water :math:`\sf K`
``R``                           Universal gas constant in
                                :math:`\sf J mol^{-1}K{^-1}`
``molar_mass_dry_air``          Molar mass for dry air in
                                :math:`\sf kg\,mol^{-1}`
``molar_mass_water``            Molar mass for water vapor in
                                :math:`\sf kg\,mol^{-1}`
``gas_constant_dry_air``        Gas constant for dry air in
                                :math:`\sf J K^{-1} kg^{-1}`
``gas_constant_water_vapor``    Gas constant for water vapor in
                                :math:`\sf J K^{-1} kg^{-1}`
``isobaric_mass_heat_capacity`` Specific heat capacity in
                                :math:`\sf J kg^{-1} K^{-1}`
``heat_of_vaporization``        Heat of vaporization in :math:`\sf J kg{^-1}`
=============================== ==============================================

Mathematical constants
======================

==========  ============
``golden``  Golden ratio
==========  ============

SI prefixes
===========

=========  ================
``yotta``  :math:`10^{24}`
``zetta``  :math:`10^{21}`
``exa``    :math:`10^{18}`
``peta``   :math:`10^{15}`
``tera``   :math:`10^{12}`
``giga``   :math:`10^{9}`
``mega``   :math:`10^{6}`
``kilo``   :math:`10^{3}`
``hecto``  :math:`10^{2}`
``deka``   :math:`10^{1}`
``deci``   :math:`10^{-1}`
``centi``  :math:`10^{-2}`
``milli``  :math:`10^{-3}`
``micro``  :math:`10^{-6}`
``nano``   :math:`10^{-9}`
``pico``   :math:`10^{-12}`
``femto``  :math:`10^{-15}`
``atto``   :math:`10^{-18}`
``zepto``  :math:`10^{-21}`
=========  ================

Non-SI ratios
=============

=======  =====================================
``ppm``  :math:`10^{-6}` `parts per million`
``ppb``  :math:`10^{-9}` `parts per billion`
``ppt``  :math:`10^{-12}` `parts per trillion`
=======  =====================================

Binary prefixes
===============

=================  ==============
``kibi``, ``KiB``  :math:`2^{10}`
``mebi``, ``MiB``  :math:`2^{20}`
``gibi``           :math:`2^{30}`
``tebi``           :math:`2^{40}`
``pebi``           :math:`2^{50}`
``exbi``           :math:`2^{60}`
``zebi``           :math:`2^{70}`
``yobi``           :math:`2^{80}`
=================  ==============

=================  ==============
``KB``             :math:`10^3`
``MB``             :math:`10^6`
=================  ==============

Earth characteristics
=====================

================  =====================================
``earth_mass``    Earth mass in :math:`\sf kg`
``earth_radius``  Earth radius in :math:`\sf m`
``atm``           Standard atmosphere in :math:`\sf Pa`
================  =====================================

"""
import numpy as np
import scipy.constants as spc

# Physical constants
g = earth_standard_gravity = spc.g  # m s^-2
h = planck = spc.Planck  # J s
k = boltzmann = spc.Boltzmann  # J K^-1
c = speed_of_light = spc.speed_of_light  # m s^-1
N_A = avogadro = N = spc.Avogadro  # mol^-1
K = zero_celsius = 273.15  # Kelvin at 0 Celsius
triple_point_water = 273.16  # Triple point temperature in K
R = gas_constant = spc.gas_constant  # J mol^-1 K^-1
mu_B = spc.e * spc.Planck * (0.25 / spc.pi) / spc.m_e  # J T^-1
molar_mass_dry_air = 28.9645e-3  # kg mol^-1
# https://www.e-education.psu.edu/meteo300/node/534
molar_mass_ozone = 48.0e-3  # kg mol^-1
molar_mass_water = 18.01528e-3  # kg mol^-1
gas_constant_dry_air = R / molar_mass_dry_air  # J K^-1 kg^-1
gas_constant_water_vapor = R / molar_mass_water  # J K^-1 kg^-1
amu = spc.m_u
stefan_boltzmann_constant = 2 * np.pi**5 * k**4 / (15 * c**2 * h**3)
isobaric_mass_heat_capacity = 1003.5  # J kg^-1 K^-1
heat_of_vaporization = 2501000  # J kg^-1

# Mathematical constants
golden = golden_ratio = (1 + np.sqrt(5)) / 2

# SI prefixes
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21

# Non-SI ratios
ppm = 1e-6  # parts per million
ppb = 1e-9  # parts per billion
ppt = 1e-12  # parts per trillion

# Binary prefixes
kibi = KiB = 2**10
mebi = MiB = 2**20
gibi = 2**30
tebi = 2**40
pebi = 2**50
exbi = 2**60
zebi = 2**70
yobi = 2**80

KB = 10**3
MB = 10**6

# Earth characteristics
earth_mass = 5.97237e24  # kg
earth_radius = 6.3781e6  # m
atm = atmosphere = 101325  # Pa












# instruments
# BANDNAMES = {}
# BANDNAMES['generic'] = {'VIS006': 'VIS0.6',
#                         'VIS008': 'VIS0.8',
#                         'IR_016': 'NIR1.6',
#                         'IR_039': 'IR3.9',
#                         'WV_062': 'IR6.2',
#                         'WV_073': 'IR7.3',
#                         'IR_087': 'IR8.7',
#                         'IR_097': 'IR9.7',
#                         'IR_108': 'IR10.8',
#                         'IR_120': 'IR12.0',
#                         'IR_134': 'IR13.4',
#                         'HRV': 'HRV',
#                         'I01': 'I1',
#                         'I02': 'I2',
#                         'I03': 'I3',
#                         'I04': 'I4',
#                         'I05': 'I5',
#                         'M01': 'M1',
#                         'M02': 'M2',
#                         'M03': 'M3',
#                         'M04': 'M4',
#                         'M05': 'M5',
#                         'M06': 'M6',
#                         'M07': 'M7',
#                         'M08': 'M8',
#                         'M09': 'M9',
#                         'C01': 'ch1',
#                         'C02': 'ch2',
#                         'C03': 'ch3',
#                         'C04': 'ch4',
#                         'C05': 'ch5',
#                         'C06': 'ch6',
#                         'C07': 'ch7',
#                         'C08': 'ch8',
#                         'C09': 'ch9',
#                         'C10': 'ch10',
#                         'C11': 'ch11',
#                         'C12': 'ch12',
#                         'C13': 'ch13',
#                         'C14': 'ch14',
#                         'C15': 'ch15',
#                         'C16': 'ch16',
#                         }
# # handle arbitrary channel numbers
# for chan_num in range(1, 37):
#     BANDNAMES['generic'][str(chan_num)] = 'ch{:d}'.format(chan_num)

# # MODIS RSR files were made before 'chX' became standard in pyspectral
# BANDNAMES['modis'] = {str(chan_num): str(chan_num) for chan_num in range(1, 37)}

# BANDNAMES['avhrr-3'] = {'1': 'ch1',
#                         '2': 'ch2',
#                         '3b': 'ch3b',
#                         '3a': 'ch3a',
#                         '4': 'ch4',
#                         '5': 'ch5'}

# BANDNAMES['ahi'] = {'B01': 'ch1',
#                     'B02': 'ch2',
#                     'B03': 'ch3',
#                     'B04': 'ch4',
#                     'B05': 'ch5',
#                     'B06': 'ch6',
#                     'B07': 'ch7',
#                     'B08': 'ch8',
#                     'B09': 'ch9',
#                     'B10': 'ch10',
#                     'B11': 'ch11',
#                     'B12': 'ch12',
#                     'B13': 'ch13',
#                     'B14': 'ch14',
#                     'B15': 'ch15',
#                     'B16': 'ch16'
#                     }

# BANDNAMES['ami'] = {'VI004': 'ch1',
#                     'VI005': 'ch2',
#                     'VI006': 'ch3',
#                     'VI008': 'ch4',
#                     'NR013': 'ch5',
#                     'NR016': 'ch6',
#                     'SW038': 'ch7',
#                     'WV063': 'ch8',
#                     'WV069': 'ch9',
#                     'WV073': 'ch10',
#                     'IR087': 'ch11',
#                     'IR096': 'ch12',
#                     'IR105': 'ch13',
#                     'IR112': 'ch14',
#                     'IR123': 'ch15',
#                     'IR133': 'ch16'
#                     }

# BANDNAMES['fci'] = {'vis_04': 'ch1',
#                     'vis_05': 'ch2',
#                     'vis_06': 'ch3',
#                     'vis_08': 'ch4',
#                     'vis_09': 'ch5',
#                     'nir_13': 'ch6',
#                     'nir_16': 'ch7',
#                     'nir_22': 'ch8',
#                     'ir_38': 'ch9',
#                     'wv_63': 'ch10',
#                     'wv_73': 'ch11',
#                     'ir_87': 'ch12',
#                     'ir_97': 'ch13',
#                     'ir_105': 'ch14',
#                     'ir_123': 'ch15',
#                     'ir_133': 'ch16'
#                     }

# BANDNAMES['slstr'] = {'S1': 'ch1',
#                       'S2': 'ch2',
#                       'S3': 'ch3',
#                       'S4': 'ch4',
#                       'S5': 'ch5',
#                       'S6': 'ch6',
#                       'S7': 'ch7',
#                       'S8': 'ch8',
#                       'S9': 'ch9',
#                       'F1': 'ch7',
#                       'F2': 'ch8',
#                       }

INSTRUMENTS = {'NOAA-19': 'avhrr/3',
               'NOAA-18': 'avhrr/3',
               'NOAA-17': 'avhrr/3',
               'NOAA-16': 'avhrr/3',
               'NOAA-15': 'avhrr/3',
               'NOAA-14': 'avhrr/2',
               'NOAA-12': 'avhrr/2',
               'NOAA-11': 'avhrr/2',
               'NOAA-9': 'avhrr/2',
               'NOAA-7': 'avhrr/2',
               'NOAA-10': 'avhrr/1',
               'NOAA-8': 'avhrr/1',
               'NOAA-6': 'avhrr/1',
               'TIROS-N': 'avhrr/1',
               'Metop-A': 'avhrr/3',
               'Metop-B': 'avhrr/3',
               'Metop-C': 'avhrr/3',
               'Suomi-NPP': 'viirs',
               'NOAA-20': 'viirs',
               'EOS-Aqua': 'modis',
               'EOS-Terra': 'modis',
               'FY-3D': 'mersi-2',
               'FY-3C': 'virr',
               'FY-3B': 'virr',
               'Feng-Yun 3D': 'mersi-2',
               'Meteosat-11': 'seviri',
               'Meteosat-10': 'seviri',
               'Meteosat-9': 'seviri',
               'Meteosat-8': 'seviri',
               'FY-4A': 'agri',
               'GEO-KOMPSAT-2A': 'ami',
               'MTG-I1': 'fci'
               }
OSCAR_PLATFORM_NAMES = {'eos-2': 'EOS-Aqua',
                        'meteosat-11': 'Meteosat-11',
                        'meteosat-10': 'Meteosat-10',
                        'meteosat-9': 'Meteosat-9',
                        'meteosat-8': 'Meteosat-8'}