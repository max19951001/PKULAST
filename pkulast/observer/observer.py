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
Observation Geometry.
=====================

Provides observation geometry.

"""

from pkulast.surface import get_band_emissivity
from pkulast.config import EMIS_LIB_290
from pkulast.rtm.model import ModtranWrapper


class Observer(object):
    """ sensor observation geometry
	"""
    def __init__(self, rsr, is_satellite, vza=0, flight_altitude=None):
        if is_satellite:
            flight_altitude = 100 # unit:km
        if flight_altitude is None:
            raise ValueError('Non-satellite-borne sensor need provide flight altitude!')
        self.filter = rsr
        self.is_satellite = is_satellite
        self.vza = vza
        self.flight_altitude = flight_altitude

    def simulate(self, profiles, delta_Ts=range(-15, 21, 5), delta_VZAs=[0, ], emis_lib=EMIS_LIB_290):
        emis = get_band_emissivity(self.filter, emis_lib=emis_lib)
        records = []
        mw = ModtranWrapper(self.filter, self.is_satellite, self.flight_altitude)

        for p in profiles:
            Ts = p.Ts
            for delta_vza in delta_VZAs:
                vza = self.vza + delta_vza
                uprad, downrad, tau = mw.correction([p, ], include_solar=True, vza=vza)
                for e in emis:
                    for delta_t in delta_Ts:
                        ts = Ts + delta_t
                        radiance = tau * ( e * self.filter.bt2r(ts) + (1-e) * downrad) + uprad
                        records.append((radiance, ts, vza, *uprad, *downrad, *tau, *e))
        return records

    def __str__(self):
        return f'Observation:\nsensor:{self.rsr}\nflight height:{self.flight_altitude}\nsatellite:{self.is_satellite}'
